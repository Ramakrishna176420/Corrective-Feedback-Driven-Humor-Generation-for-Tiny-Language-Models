import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional: set/override HF token
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_punchlines_batched(
    model_path: str,
    output_csv_path: str,
    *,
    seed: int = 42,
    test_data_path: str = "data/new_test_data_7k.csv",
    batch_size: int = 8,
    max_gen_length: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
):
    """
    Generate punchlines in batches using a LLaMA model saved at `model_path`.

    Usage:
        generate_punchlines_batched(MODEL_DIR, "./results/out.csv",
                                    seed=i, test_data_path=..., max_gen_length=50)

    Each prompt:
        "Setup: <text>\\nPunchline:"
    Only the continuation after 'Punchline:' is kept.
    """
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[Seed={seed}] Loading tokenizer/model from: {model_path}")
    # --- Load tokenizer with fallback ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )
    except Exception as e:
        print(f"Fast tokenizer failed ({e}); retrying with use_fast=False")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            trust_remote_code=True,
            use_fast=False,
            token=os.getenv("HF_TOKEN"),
        )

    # Ensure eos/pad tokens exist and are aligned
    if tokenizer.eos_token is None:
        # Common for some checkpoints
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # --- Load model (fp16 on GPU; fp32 on CPU for safety) ---
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=os.getenv("HF_TOKEN"),
    )
    # Keep generation/token IDs consistent
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    model.eval()
    print(f"Model ready on {device}")

    # --- Load & validate data ---
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")
    df = pd.read_csv(test_data_path)
    for col in ("setup", "punchline"):
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in {test_data_path}")

    setups = df["setup"].astype(str).tolist()
    gts = df["punchline"].astype(str).tolist()

    print(f"Generating {len(setups)} punchlines ...")
    results = []

    for idx in tqdm(range(0, len(setups), batch_size), desc=f"Seed {seed} - Generating"):
        batch_setups = setups[idx:idx + batch_size]
        batch_gts = gts[idx:idx + batch_size]

        prompts = [f"Setup: {s}\nPunchline:" for s in batch_setups]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
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

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for j, full_text in enumerate(decoded):
            marker = "Punchline:"
            k = full_text.rfind(marker)
            if k != -1:
                punchline_only = full_text[k + len(marker):].strip()
            else:
                # Fallback: strip prompt prefix if present
                pref = prompts[j]
                punchline_only = full_text[len(pref):].strip() if full_text.startswith(pref) else full_text.strip()

            results.append({
                "setup": batch_setups[j],
                "ground_truth_punchline": batch_gts[j],
                "generated_punchline": punchline_only,
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Save CSV (make sure dir exists even if path has no dir part) ---
    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv_path, index=False)
    print(f"[Seed={seed}] Saved: {output_csv_path}")
