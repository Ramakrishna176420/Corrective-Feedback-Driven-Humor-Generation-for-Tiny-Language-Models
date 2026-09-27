from generate_jokes import generate_punchlines_batched, generate_punchlines_bart, generate_punchlines_mistral

# mistral_gpt2
for i in range(3):
    generate_punchlines_mistral('512960/mistral_gpt2_generator_epoch_15', f'./results/mistral_gpt2_seed_{i}.csv', seed=i)