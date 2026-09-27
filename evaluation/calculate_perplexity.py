import pandas as pd
import evaluate
import numpy as np
from tabulate import tabulate

# Load evaluation metrics
perplexity_metric = evaluate.load("perplexity", module_type="metric")


def compute_perplexity(df, model_name, model_id='gpt2'):
    """
    Compute perplexity for generated punchlines.
    """
    input_texts = df['generated_punchline'].tolist()
    perplexity_results = perplexity_metric.compute(
        model_id=model_id,
        predictions=input_texts,
        add_start_token=True
    )
    return perplexity_results['mean_perplexity']


def process_seeds(seed_files, model_name):
    """
    Process and compute metrics for all seeds of a model.
    """
    print(f"Processing {model_name}...")
    metrics = {'perplexity': []}

    for seed_file in seed_files:
        print(f"  Loading file: {seed_file}")
        df = pd.read_csv(seed_file)

        # Perplexity
        perplexity = compute_perplexity(df, model_name)
        metrics['perplexity'].append(perplexity)

    # Aggregate statistics (mean and standard deviation)
    stats = {
        'perplexity_mean': np.mean(metrics['perplexity']),
        'perplexity_std': np.std(metrics['perplexity']),
    }
    return stats


def main():
    # Define seed files for each model
    models = {
        # 'GPT-2 Obj2': [f"../results/gpt2_obj2_seed_{i}.csv" for i in range(3)],
        # 'Finetuned GPT-2': [f"../results/finetuned_gpt2_seed_{i}.csv" for i in range(3)],
        # 'GPT-2 OneWord 2ndVariant': [f"../results/gpt2_oneword_2ndvariant_seed_{i}.csv" for i in range(3)],
        # 'GPT-2 Mistral': [f"../results/gpt2_mistral_seed_{i}.csv" for i in range(3)],
        # 'BART Obj2': [f"../results/bart_obj2_seed_{i}.csv" for i in range(3)],
        'Finetuned BART': [f"../results/finetuned_bart_seed_{i}.csv" for i in range(3)],
        'Mistral GPT-2': [f"../results/mistral_gpt2_seed_{i}.csv" for i in range(3)],
        'Mistral Zero-Shot': [f"../results/mistral_zero_shot_seed_{i}.csv" for i in range(3)],
    }

    # Placeholder for results
    all_results = []

    # Process each model
    for model_name, seed_files in models.items():
        print("\n" + "=" * 50)
        print(f"PROCESSING MODEL: {model_name}")
        print("=" * 50)
        stats = process_seeds(seed_files, model_name)
        print(f"  Aggregated Results for {model_name}:")
        print(stats)
        print()

        # Prepare results for tabulation
        all_results.append({
            'Model': model_name,
            'Perplexity (mean ± std)': f"{stats['perplexity_mean']:.4f} ± {stats['perplexity_std']:.4f}",
        })

    # Display results in a tabular format
    print("\n" + "=" * 50)
    print("FINAL AGGREGATED RESULTS")
    print("=" * 50)
    print(tabulate(all_results, headers="keys", tablefmt="grid"))


# Execute the main function
if __name__ == "__main__":
    main()