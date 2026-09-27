import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def generate_punchlines_zero_shot_qwen(
    model_path="Qwen/Qwen3-8B", 
    output_csv_path="filtred_qwen_zero_shot_output.csv", 
    test_data_path="data/test_data2.csv",
    seed=42,
    batch_size=32,
    max_gen_length=50,
    temperature=0.7,
    top_k=50,
    top_p=0.95
):
    """
    Generates punchlines in a zero-shot manner using Qwen. The input is a setup, and the model generates the punchline.
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer with proper configuration
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            padding_side="left",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading tokenizer with fast mode, trying slow tokenizer: {e}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            padding_side="left",
            trust_remote_code=True,
            use_fast=False
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    
    print(f"Model loaded successfully on {device}")
    
    # Load test data
    test_data = pd.read_csv(test_data_path)
    setups = test_data['setup'].tolist()
    punchlines = test_data['punchline'].tolist()
    
    print(f"Processing {len(setups)} examples with seed {seed}...")
    
    # Initialize results DataFrame
    result_df = pd.DataFrame(columns=['setup', 'ground_truth_punchline', 'generated_punchline'])
    
    # Process in batches
    for i in tqdm(range(0, len(setups), batch_size), desc=f"Generating (seed={seed})"):
        batch_setups = setups[i:i + batch_size]
        batch_punchlines = punchlines[i:i + batch_size]
        
        # Tokenize batch
        encoded_input = tokenizer(
            batch_setups,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Generate punchlines for batch
        with torch.no_grad():
            outputs = model.generate(
                **encoded_input,
                max_new_tokens=max_gen_length,
                temperature=temperature,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode generated outputs
        generated_punchlines = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Store results for batch
        batch_results = {
            'setup': batch_setups,
            'ground_truth_punchline': batch_punchlines,
            'generated_punchline': generated_punchlines
        }
        result_df = pd.concat([
            result_df, 
            pd.DataFrame(batch_results)
        ], ignore_index=True)
        
        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results
    result_df.to_csv(output_csv_path, index=False)
    print(f"Zero-shot results saved to {output_csv_path}")

# Generate with different seeds
generate_punchlines_zero_shot_qwen(output_csv_path="./results/filtred_qwen_zero_shot_seed_0.csv", seed=0)
generate_punchlines_zero_shot_qwen(output_csv_path="./results/filtred_qwen_zero_shot_seed_1.csv", seed=1)
generate_punchlines_zero_shot_qwen(output_csv_path="./results/filtred_qwen_zero_shot_seed_2.csv", seed=2)
generate_punchlines_zero_shot_qwen(output_csv_path="./results/filtred_qwen_zero_shot_seed_3.csv", seed=3)
generate_punchlines_zero_shot_qwen(output_csv_path="./results/filtred_qwen_zero_shot_seed_4.csv", seed=4)