import pandas as pd
import itertools
import random
import time
import re
import os
import argparse
from openai import OpenAI
from tabulate import tabulate


N_SAMPLES = 500
BATCH_SIZE = 100
RANDOM_SEED = 42

LLM_JUDGE_MODEL = "gpt-4.1-mini"
MAX_RETRIES = 2
REQUEST_TIMEOUT = 45
RATE_LIMIT_DELAY = 0.05

OPENAI_API_KEY = ""


OLD_MODELS_TO_EVALUATE = {
    'Finetuned GPT-2': [f"../humor_gen_oct2024/results/filtered_finetuned_gpt2_seed_{i}.csv" for i in range(1)],
    'GPT-2 KD Imitation+DPO': [f"../humor_gen_oct2024/results/filtered_gpt2_acl_dpo_seed_{i}.csv" for i in range(1)],
    'GPT-2_qwen KD Imitation+DPO': [f"../humor_gen_oct2024/results/filtered_gpt2_qwen_acl_dpo_seed_seed_{i}.csv" for i in range(1)],
    'GPT-2 KD_llama Imitation+DPO': [f"../humor_gen_oct2024/results/filtered_gpt2_llama_acl_dpo_seed_seed_{i}.csv" for i in range(1)],
    'Unfiltred_GPT-2_qwen KD Imitation+DPO': [f"../humor_gen_oct2024/results/unfiltred_gpt2_qwen_acl_dpo_seed_seed_{i}.csv" for i in range(1)],
    'Unfiltred_GPT-2_llama KD Imitation+DPO': [f"../humor_gen_oct2024/results/unfiltred_gpt2_llama_acl_dpo_seed_seed_{i}.csv" for i in range(1)],
    'Mistral Finetuned': [f"../humor_gen_oct2024/results/filtered_finetuned_mistral_seed_{i}.csv" for i in range(1)],
    'Qwen Finetuned': [f"../humor_gen_oct2024/results/filtered_finetuned_qwen_seed_{i}.csv" for i in range(1)],
    'llama Finetuned': [f"../humor_gen_oct2024/results/filtered_finetuned_llama_seed_{i}.csv" for i in range(1)],
    'Unfilted Qwen Finetuned': [f"../humor_gen_oct2024/results/unfiltred_finetuned_qwen_seed_{i}.csv" for i in range(1)],
    'Unfiltred llama Finetuned': [f"../humor_gen_oct2024/results/unfiltred_finetuned_llama_seed_{i}.csv" for i in range(1)],
    'GPT-2 KD (Finetuned Mistral)': [f"../humor_gen_oct2024/results/filtered_gpt2_finetuned_mistral_kd_seed_{i}.csv" for i in range(1)],
    'GPT-2 KD (Finetuned qwen)': [f"../humor_gen_oct2024/results/filtered_gpt2_qwen_KD_seed_{i}.csv" for i in range(1)],
    'GPT-2 KD (Finetuned llama)': [f"../humor_gen_oct2024/results/filtered_gpt2_llama_KD_seed_{i}.csv" for i in range(1)],
    'GPT-2 KD (UnfiltredFinetuned llama)': [f"../humor_gen_oct2024/results/unfiltred_gpt2_finetuned_llama_kd_seed_{i}.csv" for i in range(1)],
    'GPT-2 KD (UnfiltredFinetuned qwen)': [f"../humor_gen_oct2024/results/unfiltred_gpt2_finetuned_qwen_kd_seed_{i}.csv" for i in range(1)],
    'GPT-2 Mistral (Frozen)': [f"../humor_gen_oct2024/results/filtered_frozen_mistral_seed_{i}.csv" for i in range(1)],
    'GPT-2 llama (Frozen)': [f"../humor_gen_oct2024/results/filtered_gpt2_frozen_llama_seed_{i}.csv" for i in range(1)],
    'GPT-2 qwen (Frozen)': [f"../humor_gen_oct2024/results/filtered_gpt2_frozen_qwen_seed_{i}.csv" for i in range(1)],
    'Unfiltred GPT-2 llama (Frozen)': [f"../humor_gen_oct2024/results/unfiltred_gpt2_llama_seed_{i}.csv" for i in range(1)],
    'Unfiltred GPT-2 qwen (Frozen)': [f"../humor_gen_oct2024/results/unfiltered_gpt2_frozen_qwen_seed_{i}.csv" for i in range(1)],
    'GPT-2 KD (Imitation)': [f"../humor_gen_oct2024/results/filtered_gpt2_imitation_seed_{i}.csv" for i in range(1)],
    'GPT-2_qwen_KD (Imitation)': [f"../humor_gen_oct2024/acl_paper_and_kd/outputs/filtered_gpt2_imitation_phase_qwen_output_seed_{i}.csv" for i in range(1)],
    'GPT-2_llama_KD (Imitation)': [f"../humor_gen_oct2024/acl_paper_and_kd/outputs/filtered_gpt2_imitation_phase_llama_output_seed_{i}.csv" for i in range(1)],
    'Unfiltred_GPT-2_qwen_KD (Imitation)': [f"../humor_gen_oct2024/acl_paper_and_kd/outputs/unfiltred_gpt2_imitation_phase_qwen_output_seed_{i}.csv" for i in range(1)],
    'Unfiltred_GPT-2_llama_KD (Imitation)': [f"../humor_gen_oct2024/acl_paper_and_kd/outputs/unfiltred_gpt2_imitation_phase_llama_output_seed_{i}.csv" for i in range(1)],
    'Unfiltred GPT-2 llama (LORA)': [f"../humor_gen_oct2024/results/unfiltred_gpt2_llama_seed_{i}.csv" for i in range(5)],
    'Unfiltred GPT-2 qwen (LORA)': [f"../humor_gen_oct2024/results/unfiltered_gpt2_qwen_seed_{i}.csv" for i in range(5)],
    'filtered GPT-2 llama (LORA)': [f"../humor_gen_oct2024/results/filtered_gpt2_llama_seed_{i}.csv" for i in range(5)],
    'filtered GPT-2 qwen (LORA)': [f"../humor_gen_oct2024/results/filtered_gpt2_qwen_seed_{i}.csv" for i in range(5)],
}

NEW_LORA_MODELS = {
    'GPT-2 Mistral (LoRA)': [f"../humor_gen_oct2024/results/filtered_gpt2_mistral_lora_seed_{i}.csv" for i in range(5)],
    'Unfiltred_qwen Zero-Shot': [f"../humor_gen_oct2024/results/unfiltred_qwen_zero_shot_seed_{i}.csv" for i in range(5)],
    'Unfiltred_llama Zero-Shot': [f"../humor_gen_oct2024/results/unfiltred_llama_zero_shot_seed_{i}.csv" for i in range(5)],
    'Mistral Zero-Shot': [f"../humor_gen_oct2024/results/filtered_zero_shot_mistral_seed_{i}.csv" for i in range(5)],

}

MODELS_TO_EVALUATE = {**OLD_MODELS_TO_EVALUATE, **NEW_LORA_MODELS}


def load_and_sample_data(models_dict, n_samples, seed):
    print(f"Loading and sampling up to {n_samples} jokes for each model...")
    sampled_data = {}
    for model_name, files in models_dict.items():
        if not files:
            continue

        file_path = files[0]
        try:
            df = pd.read_csv(file_path)
            required_cols = ['setup', 'generated_punchline']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Skipping {model_name} (missing columns) -> {file_path}")
                continue

            n = min(n_samples, len(df))
            sampled_data[model_name] = df.sample(n=n, random_state=seed, replace=False).reset_index(drop=True)
            print(f"   {model_name}: sampled {n} rows from {file_path}")

        except Exception as e:
            print(f"   {model_name}: failed to load {file_path} -> {e}")

    return sampled_data


def create_llm_judge_prompt(joke_a, joke_b):
    return f"""You will be given two jokes. Choose the joke that is more humorous and sounds more human-like.

Options:
1. {joke_a}
2. {joke_b}

Strictly format your answer exactly as:
Reason: <a concise 15–20 word justification>
Answer: <1 or 2>"""


def get_llm_judge_response(client, prompt):
    for attempt in range(MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=LLM_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
                timeout=REQUEST_TIMEOUT,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"     API failed (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                return None


def parse_llm_response(response_text):
    if not response_text:
        return None, None
    reason_match = re.search(r"Reason:\s*(.*)", response_text, re.IGNORECASE)
    answer_match = re.search(r"Answer:\s*([12])", response_text, re.IGNORECASE)
    reason = reason_match.group(1).strip() if reason_match else "N/A"
    answer = int(answer_match.group(1)) if answer_match else None
    return reason, answer


def build_pairs_lora_vs_old_plus_lora_vs_lora(old_names, new_names):
    # 4×26 = 104
    cross = list(itertools.product(new_names, old_names))
    # C(4,2) = 6
    within_new = list(itertools.combinations(new_names, 2))
    return cross + within_new



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-jobs", type=int, default=1, help="Total shards")
    parser.add_argument("--job-id", type=int, default=1, help="1-based shard index")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is empty. Do: export OPENAI_API_KEY='...'")

    random.seed(RANDOM_SEED)

    print(f"[Shard] num_jobs={args.num_jobs}, job_id={args.job_id}")
    client = OpenAI(api_key=OPENAI_API_KEY)

    sampled_data = load_and_sample_data(MODELS_TO_EVALUATE, N_SAMPLES, RANDOM_SEED)

    old_names = [m for m in OLD_MODELS_TO_EVALUATE.keys() if m in sampled_data]
    new_names = [m for m in NEW_LORA_MODELS.keys() if m in sampled_data]

    print(f"\nOld models loaded: {len(old_names)}")
    print(f"New LoRA models loaded: {len(new_names)}")

    if len(old_names) < 1 or len(new_names) < 2:
        print(" Need at least 1 old model and at least 2 LoRA models loaded (for LoRA-vs-LoRA pairs).")
        return

    # Build exactly 110 pairs (if 26+4 loaded)
    all_pairs = build_pairs_lora_vs_old_plus_lora_vs_lora(old_names, new_names)

    # Shard split
    model_pairs = [p for idx, p in enumerate(all_pairs) if idx % args.num_jobs == (args.job_id - 1)]

    print(f"\nTotal pairs planned (LoRA vs old + LoRA vs LoRA): {len(all_pairs)}")
    print(f"Pairs in this shard: {len(model_pairs)}")

    # Full matrix covering all models (so you can merge shards later)
    all_models = sorted(set(old_names + new_names))
    win_rate_matrix = pd.DataFrame(index=all_models, columns=all_models, dtype=float)

    # Evaluate
    for model_a_name, model_b_name in model_pairs:
        print(f"\n=== Evaluating: [{model_a_name}] vs [{model_b_name}] ===")

        wins_a, wins_b, no_votes = 0, 0, 0
        df_a = sampled_data[model_a_name]
        df_b = sampled_data[model_b_name]
        pair_samples = min(N_SAMPLES, len(df_a), len(df_b))

        log_filename = f"shard{args.job_id}_log_{model_a_name.replace(' ', '_')}_vs_{model_b_name.replace(' ', '_')}.csv"
        with open(log_filename, "w", encoding="utf-8") as log_file:
            log_file.write("sample_index,winner,reason\n")

            for start_index in range(0, pair_samples, BATCH_SIZE):
                batch_end = min(start_index + BATCH_SIZE, pair_samples)
                print(f"  Batch {start_index+1}-{batch_end}/{pair_samples}")

                for i in range(start_index, batch_end):
                    joke_a = df_a.loc[i, "generated_punchline"]
                    joke_b = df_b.loc[i, "generated_punchline"]

                    # randomize order to avoid position bias
                    if random.random() < 0.5:
                        prompt, a_is_option_1 = create_llm_judge_prompt(joke_a, joke_b), True
                    else:
                        prompt, a_is_option_1 = create_llm_judge_prompt(joke_b, joke_a), False

                    response_text = get_llm_judge_response(client, prompt)
                    reason, answer = parse_llm_response(response_text)

                    winner_name = "no-vote"
                    if answer == 1:
                        if a_is_option_1:
                            wins_a += 1; winner_name = model_a_name
                        else:
                            wins_b += 1; winner_name = model_b_name
                    elif answer == 2:
                        if a_is_option_1:
                            wins_b += 1; winner_name = model_b_name
                        else:
                            wins_a += 1; winner_name = model_a_name
                    else:
                        no_votes += 1

                    log_file.write(f'{i},"{winner_name}","{reason}"\n')
                    time.sleep(RATE_LIMIT_DELAY)

        total_votes = wins_a + wins_b
        win_a = (wins_a / total_votes * 100.0) if total_votes > 0 else 0.0
        win_b = (wins_b / total_votes * 100.0) if total_votes > 0 else 0.0

        win_rate_matrix.loc[model_a_name, model_b_name] = win_a
        win_rate_matrix.loc[model_b_name, model_a_name] = win_b

        print(f"  Final: {model_a_name} vs {model_b_name}: {win_a:.2f}% / {win_b:.2f}% (no-votes={no_votes})")

    # Diagonal = 50
    for m in all_models:
        win_rate_matrix.loc[m, m] = 50.0

    print("\nPartial Win-Rate Matrix (this shard):")
    print(tabulate(win_rate_matrix, headers="keys", tablefmt="grid", floatfmt=".2f"))

    # Save shard matrix
    shard_matrix_path = f"shard{args.job_id}_win_rate_matrix.csv"
    win_rate_matrix.to_csv(shard_matrix_path)
    print(f"\nSaved partial matrix: {shard_matrix_path}")

    # Save shard pair list
    pairs_path = f"model_pairs_shard{args.job_id}.txt"
    with open(pairs_path, "w", encoding="utf-8") as f:
        for a, b in model_pairs:
            f.write(f"{a} || {b}\n")
    print(f"Saved pair list: {pairs_path}")


if __name__ == "__main__":
    main()


