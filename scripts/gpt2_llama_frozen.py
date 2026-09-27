import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from torch.optim import AdamW
import os

from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

import warnings
warnings.filterwarnings('ignore')

# Set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

print("torch.cuda.is_available(): ", torch.cuda.is_available())

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
train_punchlines = train_data['punchline'].tolist()

val_setups = val_data['setup'].tolist()
val_punchlines = val_data['punchline'].tolist()

train_dataset = CustomDataset(train_setups, train_punchlines)
val_dataset = CustomDataset(val_setups, val_punchlines)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize generator (GPT2)
generator = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
g_tokenizer = AutoTokenizer.from_pretrained('gpt2')

g_optim = torch.optim.Adam(generator.parameters(), lr=2e-5)

# Initialize Qwen3-8B with frozen weights
model_name = "meta-llama/Meta-Llama-3-8B"  # Using Qwen2.5 as Qwen3 may not be released yet
os.environ["HF_TOKEN"] = ""

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the model with quantization config
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    use_cache=False,
    trust_remote_code=True  # Required for Qwen models
)

# Freeze all parameters
for param in base_model.parameters():
    param.requires_grad = False

# Add LoRA configuration - Updated target modules for Qwen architecture
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen uses same attention architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for k-bit training
base_model = prepare_model_for_kbit_training(base_model)

# Get PEFT model
discriminator = get_peft_model(base_model, lora_config)
discriminator.eval()  # Keep in eval mode since we're not training it

# Initialize tokenizer for Qwen
d_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
# Qwen models typically have pad_token set, but if not:
if d_tokenizer.pad_token is None:
    d_tokenizer.pad_token = d_tokenizer.eos_token

def get_mod_tensors(string_tensor):
    assert isinstance(string_tensor, torch.Tensor), "Input must be a tensor"
    
    encoded = []
    for i in string_tensor:
        decoded = g_tokenizer.decode(i.unsqueeze(0))
        re_encoded = d_tokenizer.encode(decoded, return_tensors='pt')
        encoded.append(re_encoded[:, -1])

    encoded_tensor = torch.cat(encoded, dim=0)
    mod_tensors = encoded_tensor
    return mod_tensors

from tqdm import tqdm

# Initialize lists to log losses
g_losslog, d_losslog, total_losslog = [], [], []
epochs_train_g_loss, epochs_train_d_loss, epochs_train_total_loss = [], [], []
epochs_val_g_loss, epochs_val_d_loss, epochs_val_total_loss = [], [], []

num_epochs = 15
cnt = 0
log_interval = 20

for epoch in range(num_epochs):
    epoch_train_g_loss = []
    epoch_train_d_loss = []
    epoch_train_total_loss = []
    epoch_val_g_loss = []
    epoch_val_d_loss = []
    epoch_val_total_loss = []
    
    # Training Loop
    generator.train()
    discriminator.eval()  # Keep discriminator in eval mode
    
    for batch in tqdm(train_dataloader):
        try:
            torch.cuda.empty_cache()

            setups, punchlines = batch
            encoded_setups = [g_tokenizer.encode(setup, return_tensors='pt').to(device) for setup in setups]
            encoded_setups2 = [d_tokenizer.encode(setup, return_tensors='pt').to(device) for setup in setups]
            encoded_punchlines = [g_tokenizer.encode(punchline, return_tensors='pt').to(device) for punchline in punchlines]

            g_optim.zero_grad()

            g_loss = []
            g_outputs = []

            # Generator forward pass
            for setup, punchline in zip(encoded_setups, encoded_punchlines):
                g_input = torch.cat((setup, punchline), dim=1).to(device)
                g_preds = generator(input_ids=g_input, labels=g_input)

                setup_len = setup.shape[1]
                g_punchline_logits = g_preds.logits[:, setup_len:, :]

                g_loss.append(g_preds.loss)
                g_outputs.append(g_punchline_logits)

            g_loss = torch.stack(g_loss).mean()

            # Get discriminator loss (will be used for generator update)
            # Access embedding layer for Qwen model
            E = discriminator.model.model.embed_tokens.weight
            d_loss = []

            for g_output, inp2 in zip(g_outputs, encoded_setups2):
                gumbel_logits = F.gumbel_softmax(g_output, tau=1, hard=True)
                vocab_indices = torch.arange(g_tokenizer.vocab_size).to(device)
                argmax = torch.sum(gumbel_logits * vocab_indices, dim=-1)

                converted_tokens = get_mod_tensors(argmax.squeeze().long())
                disc_tokens = torch.tensor(converted_tokens).to(device)

                final = E[disc_tokens].unsqueeze(0)
                
                # Add straight-through gradient
                gumbel_mean = gumbel_logits.mean(dim=-1)
                gumbel_mean = gumbel_mean.unsqueeze(-1)
                gumbel_mean = gumbel_mean.expand_as(final)
                final = final + (gumbel_mean - gumbel_mean.detach())

                setup_tokens = torch.tensor(inp2).to(device).squeeze(dim=0)
                setup_embeddings = torch.stack([E[i, :] for i in setup_tokens]).unsqueeze(0)

                model2_embeddings = torch.cat((setup_embeddings, final), dim=1)
                labels = torch.cat((setup_tokens, disc_tokens.squeeze()), dim=0).unsqueeze(dim=0)

                with torch.no_grad():
                    output2 = discriminator(
                        inputs_embeds=model2_embeddings.half(),
                        labels=labels.long()
                    )
                
                d_loss.append(output2.loss)

            d_loss = torch.stack(d_loss).mean()
            
            # Combined loss for generator update
            total_loss = g_loss + d_loss

            # Update generator using combined loss
            total_loss.backward()
            g_optim.step()

            # Log losses
            epoch_train_g_loss.append(g_loss.item())
            epoch_train_d_loss.append(d_loss.item())
            epoch_train_total_loss.append(total_loss.item())
            g_losslog.append(g_loss.item())
            d_losslog.append(d_loss.item())
            total_losslog.append(total_loss.item())

            cnt += 1
            if cnt % log_interval == 0:
                print(f"Training: g_loss: {g_loss.item():.4f}, d_loss: {d_loss.item():.4f}, total_loss: {total_loss.item():.4f}")
        
        except Exception as e:
            print(f"Error in batch during training: {batch}. Skipping this batch. Error: {str(e)}")
    
    # Validation Loop
    generator.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            try:
                setups, punchlines = batch
                encoded_setups = [g_tokenizer.encode(setup, return_tensors='pt').to(device) for setup in setups]
                encoded_setups2 = [d_tokenizer.encode(setup, return_tensors='pt').to(device) for setup in setups]
                encoded_punchlines = [g_tokenizer.encode(punchline, return_tensors='pt').to(device) for punchline in punchlines]

                g_loss = []
                d_loss = []

                # Compute generator and discriminator losses for validation
                for setup, punchline in zip(encoded_setups, encoded_punchlines):
                    g_input = torch.cat((setup, punchline), dim=1).to(device)
                    g_preds = generator(input_ids=g_input, labels=g_input)
                    g_loss.append(g_preds.loss)

                g_loss = torch.stack(g_loss).mean()
                
                # Compute discriminator loss for validation
                g_outputs = []
                for setup, punchline in zip(encoded_setups, encoded_punchlines):
                    g_input = torch.cat((setup, punchline), dim=1).to(device)
                    g_preds = generator(input_ids=g_input)
                    setup_len = setup.shape[1]
                    g_punchline_logits = g_preds.logits[:, setup_len:, :]
                    g_outputs.append(g_punchline_logits)

                for g_output, inp2 in zip(g_outputs, encoded_setups2):
                    gumbel_logits = F.gumbel_softmax(g_output, tau=1, hard=True)
                    vocab_indices = torch.arange(g_tokenizer.vocab_size).to(device)
                    argmax = torch.sum(gumbel_logits * vocab_indices, dim=-1)

                    converted_tokens = get_mod_tensors(argmax.squeeze().long())
                    disc_tokens = torch.tensor(converted_tokens).to(device)

                    final = E[disc_tokens].unsqueeze(0)
                    setup_tokens = torch.tensor(inp2).to(device).squeeze(dim=0)
                    setup_embeddings = torch.stack([E[i, :] for i in setup_tokens]).unsqueeze(0)

                    model2_embeddings = torch.cat((setup_embeddings, final), dim=1)
                    labels = torch.cat((setup_tokens, disc_tokens.squeeze()), dim=0).unsqueeze(dim=0)

                    output2 = discriminator(
                        inputs_embeds=model2_embeddings.half(),
                        labels=labels.long()
                    )
                    d_loss.append(output2.loss)

                d_loss = torch.stack(d_loss).mean()
                total_loss = g_loss + d_loss

                epoch_val_g_loss.append(g_loss.item())
                epoch_val_d_loss.append(d_loss.item())
                epoch_val_total_loss.append(total_loss.item())
       
            except Exception as e:
                print(f"Error in batch during validation: {batch}. Skipping this batch. Error: {str(e)}")

    # Append epoch averages
    epochs_train_g_loss.append(np.mean(epoch_train_g_loss))
    epochs_train_d_loss.append(np.mean(epoch_train_d_loss))
    epochs_train_total_loss.append(np.mean(epoch_train_total_loss))
    epochs_val_g_loss.append(np.mean(epoch_val_g_loss))
    epochs_val_d_loss.append(np.mean(epoch_val_d_loss))
    epochs_val_total_loss.append(np.mean(epoch_val_total_loss))

    print(f"Epoch {epoch + 1}/{num_epochs}, \n"
          f"Train Generator Loss: {np.mean(epoch_train_g_loss):.4f},\n"
          f"Train Discriminator Loss: {np.mean(epoch_train_d_loss):.4f},\n"
          f"Train Total Loss: {np.mean(epoch_train_total_loss):.4f},\n"
          f"Val Generator Loss: {np.mean(epoch_val_g_loss):.4f},\n"
          f"Val Discriminator Loss: {np.mean(epoch_val_d_loss):.4f},\n"
          f"Val Total Loss: {np.mean(epoch_val_total_loss):.4f}")

    # Save generator model after each epoch
    generator.save_pretrained(f"./models/filterd_gpt2_frozen_llama_generator_epoch_{epoch + 1}")

# Save loss logs
for name, data in [
    ('generator', g_losslog),
    ('discriminator', d_losslog),
    ('total', total_losslog),
    ('epochs_train_g', epochs_train_g_loss),
    ('epochs_train_d', epochs_train_d_loss),
    ('epochs_train_total', epochs_train_total_loss),
    ('epochs_val_g', epochs_val_g_loss),
    ('epochs_val_d', epochs_val_d_loss),
    ('epochs_val_total', epochs_val_total_loss)
]:
    with open(f'logs/gpt2_frozen_qwen_{name}_loss.txt', 'w') as f:
        for item in data:
            f.write("%s\n" % item)