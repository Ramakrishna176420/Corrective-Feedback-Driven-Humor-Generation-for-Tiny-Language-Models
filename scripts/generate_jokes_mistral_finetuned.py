from generate_jokes import generate_punchlines_batched, generate_punchlines_bart, generate_punchlines_mistral

# mistral_gpt2
for i in range(3):
    generate_punchlines_mistral('540182/mistral-7b-finetuned', f'./results/mistral_finetuned_seed_{i}.csv', seed=i)