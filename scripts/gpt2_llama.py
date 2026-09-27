import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
#from transformers import AutoTokenizer, GPT2LMHeadModel, AdamW
from transformers import AutoTokenizer, GPT2LMHeadModel
from torch.optim import AdamW
import os

from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)

import warnings
warnings.filterwarnings('ignore')

# Set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

print("torch.cuda.is_available(): ", torch.cuda.is_available()) # torch

os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

train_data = pd.read_csv('../humor_gen_oct2024/data/train_set_20k.csv')
val_data = pd.read_csv('../humor_gen_oct2024/data/validation_set_5k.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, setups, punchlines):
        self.setups = setups
        self.punchlines = punchlines

    def __len__(self):
        return len(self.setups)

    def __getitem__(self, idx):
        setup = self.setups[idx]
        punchline = self.punchlines[idx]
        return setup, punchline
    
train_setups = train_data['setup'].tolist()
train_punchlines  = train_data['punchline'].tolist()

val_setups = val_data['setup'].tolist()
val_punchlines = val_data['punchline'].tolist()

train_dataset = CustomDataset(train_setups, train_punchlines)
val_dataset = CustomDataset(val_setups, val_punchlines)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

os.environ["HF_TOKEN"] = ""

generator = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
g_tokenizer = AutoTokenizer.from_pretrained('gpt2')

g_optim = torch.optim.Adam(generator.parameters(), lr=2e-5)

# Chose the base model you want
model_name = "meta-llama/Meta-Llama-3-8B"
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

#Load the model and quantize it
model = AutoModelForCausalLM.from_pretrained(
          model_name,
          # device_map={"": 0}, #device_map="auto" will cause a problem in the training,
          device_map="auto",
          # device_map={"": torch.device('cpu')},
          use_cache=False
)

# Add LoRA to make training more efficient
config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
discriminator = get_peft_model(model, config)
discriminator.train()

d_tokenizer = tokenizer

d_optim = torch.optim.Adam(discriminator.parameters(), lr=2e-5)

def get_mod_tensors(string_tensor):
    # Ensure input is a tensor
    assert isinstance(string_tensor, torch.Tensor), "Input must be a tensor"
    
    encoded = []
    for i in string_tensor:
        # print(i)
        # Decode and encode while keeping computations differentiable
        # Assume g_tokenizer and d_tokenizer support tensor operations
        decoded = g_tokenizer.decode(i.unsqueeze(0))  # Ensure compatibility with tokenizers
        # print(decoded)
        re_encoded = d_tokenizer.encode(decoded, return_tensors='pt')
        # encoded.append(re_encoded[: ,1:]) # first token is <s>, skip that
        encoded.append(re_encoded[: ,-1])

    # Concatenate tensors to avoid breaking the graph
    # print(encoded)
    # for i in encoded:
        # print(d_tokenizer.decode(i.squeeze()))
    encoded_tensor = torch.cat(encoded, dim=0)

    # Extract the last token from each sequence while maintaining differentiability
    # mod_tensors = encoded_tensor[:, -1]
    mod_tensors = encoded_tensor

    return mod_tensors

from tqdm import tqdm

# Initialize lists to log losses
d_losslog, g_losslog, losslog = [], [], []
epochs_train_g_loss, epochs_train_d_loss, epochs_train_total_loss = [], [], []
epochs_val_g_loss, epochs_val_d_loss, epochs_val_total_loss = [], [], []

num_epochs = 10
cnt = 0
log_interval = 20  # Log every 20 steps

for epoch in range(num_epochs):
    # Reset the epoch loss lists for each epoch
    epoch_train_g_loss = []
    epoch_train_d_loss = []
    epoch_train_total_loss = []
    epoch_val_g_loss = []
    epoch_val_d_loss = []
    epoch_val_total_loss = []
    
    # Training Loop
    generator.train()
    discriminator.train()
    
    for batch in tqdm(train_dataloader):
        try:
            torch.cuda.empty_cache()

            setups, punchlines = batch
            # print(setups, punchlines)

            encoded_setups = [g_tokenizer.encode(setup, return_tensors='pt').to(device) for setup in setups]
            encoded_setups2 = [d_tokenizer.encode(setup, return_tensors='pt').to(device) for setup in setups]
            encoded_punchlines = [g_tokenizer.encode(punchline, return_tensors='pt').to(device) for punchline in punchlines]
            encoded_punchlines2 = [d_tokenizer.encode(punchline, return_tensors='pt').to(device) for punchline in punchlines]

            # Clear gradients at the start
            g_optim.zero_grad()
            d_optim.zero_grad()

            g_loss = []
            g_outputs = []

            # Forward pass through generator
            for setup, punchline in zip(encoded_setups, encoded_punchlines):

                g_input = torch.cat((setup, punchline), dim=1).to(device)
                g_preds = generator(input_ids = g_input, labels = g_input)

                setup_len = setup.shape[1]
                g_punchline_logits = g_preds.logits[:, setup_len:, :]

                g_loss.append(g_preds.loss)
                g_outputs.append(g_punchline_logits)

            g_loss = torch.stack(g_loss).mean()

            E = discriminator.model.model.embed_tokens.weight
            d_loss = []

            for g_output, inp2 in zip(g_outputs, encoded_setups2):

                gumbel_logits = F.gumbel_softmax(g_output, tau=1, hard=True)
                # print("gumbel_logits.shape: ", gumbel_logits.shape) # [punchline_len, vocab_size]

                # Get discrete tokens for labels using weighted sum
                vocab_indices = torch.arange(g_tokenizer.vocab_size).to(device)
                argmax = torch.sum(gumbel_logits * vocab_indices, dim=-1)

                # Convert discrete tokens to discriminator space for labels
                converted_tokens = get_mod_tensors(argmax.squeeze().long())
                disc_tokens = torch.tensor(converted_tokens).to(device)

                 # Get embeddings in discriminator space
                final = E[disc_tokens].unsqueeze(0)

                # Add straight-through gradient
                gumbel_mean = gumbel_logits.mean(dim=-1)  # [1, punchline_len]
                gumbel_mean = gumbel_mean.unsqueeze(-1)  # [1, punchline_len, 1]
                gumbel_mean = gumbel_mean.expand_as(final)  # [1, punchline_len, embedding_dim]

                # Now the shapes match for addition
                final = final + (gumbel_mean - gumbel_mean.detach())

                # Process setup tokens
                setup_tokens = torch.tensor(inp2).to(device).squeeze(dim=0)
                setup_embeddings = torch.stack([E[i, :] for i in setup_tokens]).unsqueeze(0)

                # Combine setup and generated embeddings
                model2_embeddings = torch.cat((setup_embeddings, final), dim=1)

                # Create labels by combining setup and converted tokens
                labels = torch.cat((setup_tokens, disc_tokens.squeeze()), dim=0).unsqueeze(dim=0)

                # Forward pass through discriminator
                output2 = discriminator(
                    inputs_embeds=model2_embeddings.half(),
                    labels=labels.long()
                )

                d_loss.append(output2.loss)

            d_loss = torch.stack(d_loss).mean()

            loss = g_loss + d_loss

            # Backpropagation and optimizer steps for the generator
            loss.backward()
            g_optim.step()

            epoch_train_g_loss.append(g_loss.item())  # Log generator loss
            epoch_train_d_loss.append(d_loss.item())  # Log discriminator loss
            epoch_train_total_loss.append(loss.item())  # Log total loss
            g_losslog.append(g_loss.item())
            d_losslog.append(d_loss.item())
            losslog.append(loss.item())

            cnt += 1
            if cnt % log_interval == 0:
                print(f"First loop: d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, total_loss: {loss.item()}")
                d_losslog.append(d_loss.item())
                g_losslog.append(g_loss.item())
                losslog.append(loss.item())
        
        
        except Exception as e:
            print(f"Error in batch during first loop: {batch}. Skipping this batch. Error: {str(e)}")

        
        try:
            d_loss2 = []
            for idx, (setup, punchline) in enumerate(zip(encoded_setups2, encoded_punchlines2)):
                d_input = torch.cat((setup, punchline), dim=1).to(device)
                d_pred = discriminator(input_ids=d_input, labels=d_input)
                d_loss2.append(d_pred.loss)

            d_loss2 = torch.stack(d_loss2).mean()

            # Backpropagation and optimizer steps for the discriminator
            d_loss2.backward()
            d_optim.step()

            epoch_train_d_loss.append(d_loss2.item())  # Log each real data discriminator loss

            if cnt % log_interval == 0:
                print(f"2nd loop: d_loss (real data): {d_loss2}")
        
        except Exception as e:
                   print(f"Error in batch during second loop: {batch}. Skipping this batch. Error: {str(e)}")
        
        # at end of batch
        del g_loss, d_loss, g_outputs
        torch.cuda.empty_cache()
    
    # Validation Loop
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            try:
                setups, punchlines = batch
                encoded_setups = [g_tokenizer.encode(setup, return_tensors='pt').to(device) for setup in setups]
                encoded_punchlines = [g_tokenizer.encode(punchline, return_tensors='pt').to(device) for punchline in punchlines]
                encoded_setups2 = [d_tokenizer.encode(setup, return_tensors='pt').to(device) for setup in setups]
                encoded_punchlines2 = [d_tokenizer.encode(punchline, return_tensors='pt').to(device) for punchline in punchlines]

                g_loss = []
                d_loss = []

                # Validation forward pass
                for setup, punchline in zip(encoded_setups, encoded_punchlines):
                    g_input = torch.cat((setup, punchline), dim=1).to(device)
                    g_preds = generator(input_ids=g_input, labels=g_input)
                    g_loss.append(g_preds.loss)

                for setup, punchline in zip(encoded_setups2, encoded_punchlines2):
                    d_input = torch.cat((setup, punchline), dim=1).to(device)
                    d_pred = discriminator(input_ids=d_input, labels=d_input)
                    d_loss.append(d_pred.loss)

                # Collecting losses for validation
                g_loss = torch.stack(g_loss).mean()
                d_loss = torch.stack(d_loss).mean()

                epoch_val_g_loss.append(g_loss.item())  # Log each validation generator loss
                epoch_val_d_loss.append(d_loss.item())  # Log each validation discriminator loss
                epoch_val_total_loss.append((g_loss + d_loss).item())  # Log total validation loss
       
            except Exception as e:
                print(f"Error in batch during validation loop: {batch}. Skipping this batch. Error: {str(e)}")


    print(f"Epoch {epoch + 1}/{num_epochs}, \n"
      f"Train Generator Loss: {np.mean(epoch_train_g_loss):.4f},\n "  # Use mean instead of whole list
      f"Train Discriminator Loss: {np.mean(epoch_train_d_loss):.4f}, \n"
      f"Train Total Loss: {np.mean(epoch_train_total_loss):.4f}, \n"
      f"Val Generator Loss: {np.mean(epoch_val_g_loss):.4f}, \n"
      f"Val Discriminator Loss: {np.mean(epoch_val_d_loss):.4f}, \n"
      f"Val Total Loss: {np.mean(epoch_val_total_loss):.4f}")

    # Save models after each epoch
    generator.save_pretrained(f"./models/filtred_gpt2_llama_generator_epoch_{epoch + 1}")
    discriminator.save_pretrained(f"./models/filtred_gpt2_llama_discriminator_epoch_{epoch + 1}")
    
# Save loss logs to files
with open('logs/filtred_gpt2_llama_generator_losslog.txt', 'w') as f:
    for item in g_losslog:
        f.write("%s\n" % item)

with open('logs/filtred_gpt2_llama_discriminator_losslog.txt', 'w') as f:
    for item in d_losslog:
        f.write("%s\n" % item)

with open('logs/filtred_gpt2_llama_total_losslog.txt', 'w') as f:
    for item in losslog:
        f.write("%s\n" % item)

with open('logs/filtred_gpt2_llama_epochs_train_g_loss.txt', 'w') as f:
    for item in epochs_train_g_loss:
        f.write("%s\n" % item)

with open('logs/filtred_gpt2_llama_epochs_train_d_loss.txt', 'w') as f:
    for item in epochs_train_d_loss:
        f.write("%s\n" % item)

with open('logs/filtred_gpt2_llama_epochs_train_total_loss.txt', 'w') as f:
    for item in epochs_train_total_loss:
        f.write("%s\n" % item)

with open('logs/gpt2_llama_epochs_val_g_loss.txt', 'w') as f:
    for item in epochs_val_g_loss:
        f.write("%s\n" % item)

with open('logs/filtred_gpt2_llama_epochs_val_d_loss.txt', 'w') as f:
    for item in epochs_val_d_loss:
        f.write("%s\n" % item)

with open('logs/filtred_gpt2_llama_epochs_val_total_loss.txt', 'w') as f:
    for item in epochs_val_total_loss:
        f.write("%s\n" % item)