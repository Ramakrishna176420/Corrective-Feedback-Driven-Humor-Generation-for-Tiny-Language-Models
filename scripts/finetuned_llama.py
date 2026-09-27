# qwen_qlora_sft.py
import os
import random
import numpy as np
import pandas as pd
import torch
import warnings
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

warnings.filterwarnings("ignore")


MODEL_NAME   = "meta-llama/Meta-Llama-3-8B"     
USE_CHAT_TEMPLATE = False          
MAX_LENGTH  = 512

TRAIN_CSV   = "../humor_gen_oct2024/data/train_set_20k.csv"
VAL_CSV     = "../humor_gen_oct2024/data/validation_set_5k.csv"
OUT_DIR     = "models/llama_finetuned_filtred"

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)


def load_data(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    if USE_CHAT_TEMPLATE:
        
        train_df["setup"] = train_df["setup"].astype(str)
        train_df["punchline"] = train_df["punchline"].astype(str)
        val_df["setup"] = val_df["setup"].astype(str)
        val_df["punchline"] = val_df["punchline"].astype(str)
        train_df["text"] = train_df["setup"] + "\n" + train_df["punchline"]
        val_df["text"]   = val_df["setup"] + "\n" + val_df["punchline"]
    else:
        
        def fmt(row):
            return f"Setup: {row['setup']}\nPunchline: {row['punchline']}"
        train_df["text"] = train_df.apply(fmt, axis=1)
        val_df["text"]   = val_df.apply(fmt, axis=1)

    train_ds = Dataset.from_pandas(train_df[["text"]], preserve_index=False)
    val_ds   = Dataset.from_pandas(val_df[["text"]],   preserve_index=False)
    return train_ds, val_ds

def init_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def tokenize_data(ds, tok, max_length=512):
    if USE_CHAT_TEMPLATE:
        def fn(examples):
           
            prompts = []
            for t in examples["text"]:
                # t = "setup\npunchline"; we want model to learn to continue after "Punchline:"
                # Train input: "Setup: <setup>\nPunchline: <punchline>"
                # We still construct a user message prompting punchline completion.
                setup, _, punchline = t.partition("\n")
                msg = [{"role":"user","content": f"Setup: {setup}\nPunchline:"}]
                prompt = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
                # Supervised target includes gold continuation after the template prompt
                prompts.append(prompt + " " + punchline)
            out = tok(
                prompts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            return out
    else:
        def fn(examples):
            out = tok(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            return out

    return ds.map(fn, batched=True, remove_columns=ds.column_names)


def init_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
    )

    base = prepare_model_for_kbit_training(base)

    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ],
    )

    
    present = set()
    for n, _ in base.named_modules():
        if n.endswith(("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")):
            present.add(n.split(".")[-1])
    
    if not present:
        lora_config.target_modules = ["q_proj","k_proj","v_proj","o_proj"]

    model = get_peft_model(base, lora_config)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


print("Loading tokenizer...")
tokenizer = init_tokenizer(MODEL_NAME)

print("Loading and preparing datasets...")
train_ds, val_ds = load_data(TRAIN_CSV, VAL_CSV)

print("Tokenizing datasets...")
train_tok = tokenize_data(train_ds, tokenizer, max_length=MAX_LENGTH)
val_tok   = tokenize_data(val_ds,   tokenizer, max_length=MAX_LENGTH)

print("Initializing model...")
model = init_model(MODEL_NAME)


training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=10,                     
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    eval_strategy="steps",            
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    warmup_steps=200,
    learning_rate=2e-4,                     
    bf16=True,                              #
    fp16=False,
    logging_steps=20,
    report_to="tensorboard",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    weight_decay=0.01,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

print("Saving model & tokenizer...")
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("Training completed!")