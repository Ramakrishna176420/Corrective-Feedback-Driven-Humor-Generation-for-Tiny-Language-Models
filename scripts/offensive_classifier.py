import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

# Optional: Set your Hugging Face token
os.environ["HF_TOKEN"] = ""

# Model and tokenizer
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', token=os.environ["HF_TOKEN"])
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", token=os.environ["HF_TOKEN"])
model.eval()

# Prompt template as in your image
def make_prompt(setup, punchline):
    return f"""Offensive language refers to words or expressions intended to insult, harm, or belittle individuals or groups.
This includes:
· Insulting, derogatory, or combative language
· Discriminatory language based on race, religion, gender, or other personal attributes
· Profanity, violence, or vulgar expressions
· Sarcastic or harmful humor
Based on the above definition, determine if the following sentence is offensive language. Return only “yes” or “no”.

Input: {setup} {punchline}
Output:"""

# Custom Dataset class for inference
class OffensiveDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prompt = make_prompt(row["setup"], row["punchline"])
        return prompt

# Classify LLM outputs
def classify_outputs(outputs):
    labels = []
    for out in outputs:
        text = out.strip().lower()
        if text.startswith("yes"):
            labels.append("yes")
        elif text.startswith("no"):
            labels.append("no")
        else:
            labels.append("unknown")
    return labels

# Inference in batches
def run_batched_inference(df, batch_size=8, max_new_tokens=5):
    dataset = OffensiveDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_outputs = []
    for prompts in tqdm(dataloader, desc="Running batched inference"):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        all_outputs.extend(classify_outputs(decoded))
    return all_outputs

# Full processing function
def process_file(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df["llm_label"] = run_batched_inference(df)
    df.to_csv(output_csv, index=False)
    print(f"Filtered LLM-labeled file saved as {output_csv}")

# Run the script for your datasets
process_file("data/train_data.csv", "data/train_data_filtered.csv")
process_file("data/test_data2.csv", "data/test_data2_filtered.csv")

