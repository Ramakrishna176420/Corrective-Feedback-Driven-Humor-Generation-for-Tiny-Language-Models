import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    AdamW,
    BitsAndBytesConfig,
    GenerationConfig
)
from tqdm import tqdm

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


MODEL_SAVE_PATH = "../humor_gen_oct2024/models/kd_gpt2_mistral"   
DATA_PATH = "../humor_gen_oct2024/data"             
LOG_PATH = "../humor_gen_oct2024/logs"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)


train_df = pd.read_csv(f"{DATA_PATH}/train_set_20k.csv")
val_df = pd.read_csv(f"{DATA_PATH}/validation_set_5k.csv")


class CustomDataset(Dataset):
    def __init__(self, setups):
        self.setups = setups

    def __len__(self):
        return len(self.setups)

    def __getitem__(self, idx):
        return self.setups[idx]

train_dataset = CustomDataset(train_df["setup"].tolist())
val_dataset = CustomDataset(val_df["setup"].tolist())

BATCH_SIZE = 4
train_loader = DataLoader(train_dataset, batch_size=BATCH_e=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

student = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
student_tokenizer = AutoTokenizer.from_pretrained("gpt2")
student_tokenizer.pad_token = student_tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
teacher = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False
).eval()

teacher_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

gen_config = GenerationConfig(
    num_beams=1,
    num_return_sequences=1,  
    do_sample=True,
    temperature=0.7,
    pad_token_id=teacher_tokenizer.eos_token_id,
    max_new_tokens=64
)

def generate_teacher_texts(setups_batch):
    """
    Each batch is just a list of `setup` strings.
    We'll force teacher to return exactly 1 generation per setup.
    """
    inputs = teacher_tokenizer(
        setups_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = teacher.generate(
            **inputs,
            generation_config=gen_config
        )

    decoded = teacher_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # debug: see if #setups == #decoded
    # print(f"[DEBUG] #setups={len(setups_batch)}, #decoded={len(decoded)}")
    return decoded

def tokenize_for_student(texts):
    return student_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

EPOCHS = 4  # shorter for example
LR = 2e-5
optimizer = AdamW(student.parameters(), lr=LR)

batch_losses = []
epoch_train_loss = []
epoch_val_loss = []

for epoch in range(EPOCHS):
    student.train()
    total_train_loss = 0.0

    for batch_idx, setups_batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        # A) Teacher => generated_texts
        teacher_texts = generate_teacher_texts(setups_batch)  # list of length BATCH_SIZE
        # debug: print shapes:
        # print(f"[DEBUG] #teacher_texts={len(teacher_texts)} (should be {len(setups_batch)})")

        # B) Student sees teacher_text as input & label for LM
        student_batch = tokenize_for_student(teacher_texts)
        # debug shapes:
        # print(f"[DEBUG] student_batch input shape: {student_batch['input_ids'].shape}")

        # C) Forward pass: we train GPT-2 to replicate teacher_text
        outputs = student(**student_batch, labels=student_batch["input_ids"])
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        batch_losses.append(loss.item())

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    student.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch_idx, setups_batch in enumerate(val_loader):
            teacher_texts = generate_teacher_texts(setups_batch)
            student_batch = tokenize_for_student(teacher_texts)
            val_out = student(**student_batch, labels=student_batch["input_ids"])
            total_val_loss += val_out.loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    epoch_train_loss.append(avg_train_loss)
    epoch_val_loss.append(avg_val_loss)

    print(f"Epoch {epoch+1} - train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")

    # Save model
    epoch_outdir = os.path.join(MODEL_SAVE_PATH, f"epoch_{epoch+1}")
    student.save_pretrained(epoch_outdir)

pd.DataFrame(batch_losses, columns=["batch_loss"]).to_csv(
    os.path.join(LOG_PATH, "kd_train_loss.csv"), index=False
)

pd.DataFrame({
    "epoch": list(range(1, EPOCHS + 1)),
    "train_loss": epoch_train_loss,
    "val_loss": epoch_val_loss
}).to_csv(
    os.path.join(LOG_PATH, "kd_epoch_losses.csv"), index=False
)





