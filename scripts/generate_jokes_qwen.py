from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import pandas as pd
from tqdm import tqdm
import os
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_punchlines_batched(model_path, output_csv_path, test_data_path="data/new_test_data_7k.csv",
                                seed=42, batch_size=8, max_gen_length=50, temperature=0.7, top_k=50, top_p=0.95):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Base model name from your Qwen training script
    base_model_name = "Qwen/Qwen3-8B"
    
    # Set up HF token
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    
    print(f"Loading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, token=os.getenv("HF_TOKEN"))
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    print(f"Loading base model {base_model_name} with 4-bit quantization...")
    # Use same quantization config as training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
    )
    
    print(f"Loading LoRA adapter from {model_path}...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Load test data
    print(f"Loading test data from {test_data_path}...")
    df = pd.read_csv(test_data_path)
    setups = df['setup'].tolist()
    ground_truth_punchlines = df['punchline'].tolist()
    
    results = []
    
    # Process in batches
    print(f"Generating punchlines for {len(setups)} setups...")
    for i in tqdm(range(0, len(setups), batch_size), desc="Generating punchlines"):
        batch_setups = setups[i:i+batch_size]
        batch_ground_truth = ground_truth_punchlines[i:i+batch_size]
        
        # Format inputs to match training format
        formatted_inputs = [f"Setup: {setup}\nPunchline:" for setup in batch_setups]
        
        # Tokenize batch
        inputs = tokenizer(
            formatted_inputs, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_gen_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and extract punchlines
        for j, output in enumerate(outputs):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            # Remove the input prompt to get just the punchline
            punchline = generated_text[len(formatted_inputs[j]):].strip()
            
            # Store setup, ground truth, and generated punchline
            results.append({
                'setup': batch_setups[j],
                'ground_truth_punchline': batch_ground_truth[j],
                'generated_punchline': f"{batch_setups[j]} {punchline}"
            })
    
    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
