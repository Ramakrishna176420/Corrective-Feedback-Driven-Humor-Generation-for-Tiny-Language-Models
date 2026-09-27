import numpy as np
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

def generate_punchlines_batched(
    model_path, 
    output_csv_path, 
    test_data_path='data/test_data2.csv',
    seed=42,
    batch_size=32,
    max_gen_length=50,
    temperature=0.7,
    top_k=50,
    top_p=0.95
):
    """
    Generates punchlines using a pretrained GPT-2 model with batched processing.
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    model.eval()
    
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Load test data
    test_data = pd.read_csv(test_data_path)
    setups = test_data['setup'].tolist()
    punchlines = test_data['punchline'].tolist()

    # Initialize results DataFrame
    result_df = pd.DataFrame(columns=['setup', 'ground_truth_punchline', 'generated_punchline'])

    # Process in batches
    for i in tqdm(range(0, len(setups), batch_size)):
        batch_setups = setups[i:i + batch_size]
        batch_punchlines = punchlines[i:i + batch_size]

        # Tokenize batch
        encoded_input = tokenizer(
            batch_setups,
            padding=True,
            return_tensors='pt'
        )

        # Generate punchlines for batch
        with torch.no_grad():
            outputs = model.generate(
                **{k: v.to(device) for k, v in encoded_input.items()},
                max_new_tokens=max_gen_length,
                temperature=temperature,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
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
    print(f"Results saved to {output_csv_path}")
    
    
    
def generate_punchlines_mistral(
    model_path, 
    output_csv_path, 
    test_data_path='data/test_data2.csv',
    seed=42,
    batch_size=32,
    max_gen_length=50,
    temperature=0.7,
    top_k=50,
    top_p=0.95
):
    """
    Generates punchlines using a pretrained Mistral model with batched processing.
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    os.environ["HF_TOKEN"] = ""
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Load test data
    test_data = pd.read_csv(test_data_path)
    setups = test_data['setup'].tolist()
    punchlines = test_data['punchline'].tolist()

    # Initialize results DataFrame
    result_df = pd.DataFrame(columns=['setup', 'ground_truth_punchline', 'generated_punchline'])

    # Process in batches
    for i in tqdm(range(0, len(setups), batch_size)):
        batch_setups = setups[i:i + batch_size]
        batch_punchlines = punchlines[i:i + batch_size]

        # Tokenize batch
        encoded_input = tokenizer(
            batch_setups,
            padding=True,
            return_tensors='pt'
        )

        # Generate punchlines for batch
        with torch.no_grad():
            outputs = model.generate(
                **{k: v.to(device) for k, v in encoded_input.items()},
                max_new_tokens=max_gen_length,
                temperature=temperature,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
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
    print(f"Results saved to {output_csv_path}")
    

def generate_punchlines_zero_shot_mistral_instruct(
    model_path="mistralai/Mistral-7B-Instruct-v0.1", 
    output_csv_path="zero_shot_output.csv", 
    test_data_path="data/test_data2.csv",
    seed=42,
    batch_size=32,
    max_gen_length=50,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    token="HF_TOKEN"
):
    """
    Generates punchlines in a zero-shot manner using Mistral-7B-Instruct. 
    The input is a setup, and the model generates the punchline using proper instruction formatting.
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, token=token).to(device)
    model.eval()
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    # Load test data
    test_data = pd.read_csv(test_data_path)
    setups = test_data['setup'].tolist()
    punchlines = test_data['punchline'].tolist()
    
    # Initialize results DataFrame
    result_df = pd.DataFrame(columns=['setup', 'ground_truth_punchline', 'generated_punchline'])
    
    # Process in batches
    for i in tqdm(range(0, len(setups), batch_size)):
        batch_setups = setups[i:i + batch_size]
        batch_punchlines = punchlines[i:i + batch_size]
        
        # Format inputs with Mistral Instruct template
        formatted_inputs = []
        for setup in batch_setups:
            instruction = f"<s>[INST] You are given a joke setup. Generate a humorous punchline that completes the joke.\n\nSetup: {setup}\n\nPunchline: [/INST]"
            formatted_inputs.append(instruction)
        
        # Tokenize batch
        encoded_input = tokenizer(
            formatted_inputs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=70  # Add max_length to prevent overly long inputs
        )
        
        # Generate punchlines for batch
        with torch.no_grad():
            outputs = model.generate(
                **{k: v.to(device) for k, v in encoded_input.items()},
                max_new_tokens=max_gen_length,
                temperature=temperature,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode generated outputs and extract only the generated part
        generated_punchlines = []
        for j, output in enumerate(outputs):
            # Get the length of the input to extract only the generated part
            input_length = encoded_input['input_ids'][j].shape[0]
            generated_tokens = output[input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up the generated text (remove any residual formatting)
            generated_text = generated_text.strip()
            if generated_text.startswith('Punchline:'):
                generated_text = generated_text[10:].strip()
            
            generated_punchlines.append(generated_text)
        
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
    print(f"Zero-shot Mistral Instruct results saved to {output_csv_path}")
    
    return result_df


def generate_punchlines_mistral_corrected(
    model_path,
    output_csv_path,
    test_data_path='data/test_data2.csv',
    seed=42,
    batch_size=16, # Reduced batch size for stability with larger models
    max_gen_length=50,
    temperature=0.7,
    top_k=50,
    top_p=0.95
):
    """
    Generates punchlines using a fine-tuned Mistral model with corrected
    prompt formatting and output cleaning.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load fine-tuned model and tokenizer
    # The tokenizer should be loaded from the fine-tuned model directory
    # to ensure it has the same settings (e.g., pad token) used in training.
    
     # Load model and tokenizer
    os.environ["HF_TOKEN"] = ""
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    # Ensure pad token is set (good practice, though should be saved with model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load test data
    test_data = pd.read_csv(test_data_path)
    setups = test_data['setup'].tolist()
    punchlines = test_data['punchline'].tolist()

    # Initialize results DataFrame
    result_df = pd.DataFrame(columns=['setup', 'ground_truth_punchline', 'generated_punchline'])

    print("Generating punchlines with corrected logic...")
    # Process in batches
    for i in tqdm(range(0, len(setups), batch_size)):
        batch_setups = setups[i:i + batch_size]
        batch_punchlines = punchlines[i:i + batch_size]

        # --- KEY CHANGE 1: Format the prompts to match the training data ---
        # The model was trained on "Setup: ...\nPunchline: ...", so we prompt it
        # with the first part to encourage it to complete the punchline.
        formatted_prompts = [f"Setup: {setup}\nPunchline:" for setup in batch_setups]

        # Tokenize the correctly formatted batch of prompts
        encoded_input = tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors='pt',
            truncation=True,
            max_length=256 # Add max_length to avoid overly long setups
        )

        # Generate punchlines for the batch
        with torch.no_grad():
            outputs = model.generate(
                **{k: v.to(device) for k, v in encoded_input.items()},
                max_new_tokens=max_gen_length,
                temperature=temperature,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1
            )

        # Decode the full generated text (which includes the prompt)
        full_decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # --- KEY CHANGE 2: Clean the output to isolate the punchline ---
        # We remove the prompt we gave the model from its output.
        cleaned_punchlines = [
            full_output[len(prompt):].strip()
            for full_output, prompt in zip(full_decoded_outputs, formatted_prompts)
        ]

        # Store results for the batch in the desired clean format
        batch_results = {
            'setup': batch_setups, # The original, clean setup
            'ground_truth_punchline': batch_punchlines,
            'generated_punchline': cleaned_punchlines # The new, clean punchline
        }
        result_df = pd.concat([
            result_df,
            pd.DataFrame(batch_results)
        ], ignore_index=True)

    # Save results to a clean CSV
    result_df.to_csv(output_csv_path, index=False)
    print(f"\nClean results saved to {output_csv_path}")