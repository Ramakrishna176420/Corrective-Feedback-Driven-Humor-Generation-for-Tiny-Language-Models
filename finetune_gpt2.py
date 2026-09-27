# Code for FT-GPT2

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

import warnings
warnings.filterwarnings('ignore')

# Set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
train_data = pd.read_csv('data/train_set_20k.csv')
val_data = pd.read_csv('data/validation_set_5k.csv')
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

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

optim = torch.optim.Adam(model.parameters(), lr=2e-5) 
num_epochs = 15
train_loss_log = [] # after every log_interval iterations
epochs_train_loss = []
epochs_val_loss = []

cnt = 0
log_interval = 20

for epoch in range(num_epochs):
    
    # Training Loop
    curr_train_loss = []
    model.train()
    for batch in train_dataloader:
        setups, punchlines = batch

        encoded_setups = [tokenizer.encode(setup, return_tensors="pt") for setup in setups]
        encoded_punchlines = [tokenizer.encode(punchline, return_tensors="pt") for punchline in punchlines]

        loss = []
        outputs = []

        for setup, punchline in zip(encoded_setups, encoded_punchlines):

            input = torch.cat((setup, punchline), dim=1).to(device)
            preds = model(input, labels=input)
            loss.append(preds.loss)

        loss = torch.stack(loss).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        cnt += 1
        if cnt % log_interval == 0:
            print(f"Batch {cnt}, Training Loss: {loss.item()}")
            train_loss_log.append(loss.item())

        curr_train_loss.append(loss.item())

    # Logging the average training loss for the epoch
    avg_train_loss = np.mean(curr_train_loss)
    epochs_train_loss.append(avg_train_loss)
    print(f"Epoch {epoch+1}/{num_epochs} Training Loss: {avg_train_loss}")

    # Validation Loop
    curr_val_loss = []
    model.eval()
    with torch.no_grad():  # Disable gradient calculation during validation
        for batch in val_dataloader:
            setups, punchlines = batch

            encoded_setups = [tokenizer.encode(setup, return_tensors="pt") for setup in setups]
            encoded_punchlines = [tokenizer.encode(punchline, return_tensors="pt") for punchline in punchlines]

            loss = []
            for setup, punchline in zip(encoded_setups, encoded_punchlines):

                input = torch.cat((setup, punchline), dim=1).to(device)
                preds = model(input, labels=input)
                loss.append(preds.loss)

            loss = torch.stack(loss).mean()
            curr_val_loss.append(loss.item())

    # Logging the average validation loss for the epoch
    avg_val_loss = np.mean(curr_val_loss)
    epochs_val_loss.append(avg_val_loss)
    print(f"Epoch {epoch+1}/{num_epochs} Validation Loss: {avg_val_loss}")

    # Save model at the end of each epoch
    model.save_pretrained(f'models/finetuned_gpt2_epoch_{epoch+1}')
    
with open('logs/finetune_gpt2_train_loss_log.txt', 'w') as f:
    for item in train_loss_log:
        f.write("%s\n" % item)

with open('logs/finetune_gpt2_epochs_train_loss.txt', 'w') as f:
    for item in epochs_train_loss:
        f.write("%s\n" % item)

with open('logs/finetune_gpt2_epochs_val_loss.txt', 'w') as f:
    for item in epochs_val_loss:
        f.write("%s\n" % item)