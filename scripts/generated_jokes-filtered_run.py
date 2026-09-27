from generate_jokes import generate_punchlines_batched, generate_punchlines_mistral_corrected, generate_punchlines_zero_shot_mistral_instruct

# Finetuned GPT-2
for i in range(5):
    generate_punchlines_batched('906784/finetuned_gpt2_epoch_15', f'./results/filtered_finetuned_gpt2_seed_{i}.csv', 
                                seed = i, test_data_path = 'data/new_test_data_7k.csv', max_gen_length = 50)
    
    
# GPT-2 + Frozen Mistral
for i in range(5):
    generate_punchlines_batched('907321/gpt2_frozen_mistral_generator_epoch_15', f'./results/filtered_frozen_mistral_seed_{i}.csv', 
                                seed = i, test_data_path = 'data/new_test_data_7k.csv', max_gen_length = 50)
    
    
# GPT-2 + Mistral(LoRA)
for i in range(5):
    generate_punchlines_batched('909880/gpt2_mistral_generator_epoch_15', f'./results/filtered_gpt2_mistral_lora_seed_{i}.csv', 
                                seed = i, test_data_path = 'data/new_test_data_7k.csv', max_gen_length = 50)
    
    
# ACL: GPT-2 Imitation Phase
for i in range(5):
    generate_punchlines_batched('acl_paper_and_kd/models/filtered_gpt2_imitation_phase_epoch_15', f'./results/filtered_gpt2_imitation_seed_{i}.csv', 
                                seed = i, test_data_path = 'data/new_test_data_7k.csv', max_gen_length = 50)
    
    
# ACL: GPT-2 Imitation Phase + DPO
for i in range(5):
    generate_punchlines_batched('acl_paper_and_kd/models/filtered_gpt2_dpo_final_trl/final_checkpoint', f'./results/filtered_gpt2_acl_dpo_seed_{i}.csv', 
                                seed = i, test_data_path = 'data/new_test_data_7k.csv', max_gen_length = 50)

    
# GPT-2 Finetuned Mistral KD
for i in range(5):
    generate_punchlines_batched('acl_paper_and_kd/models/filtered_gpt2_finetuned_mistral_kd_epoch_15', f'./results/filtered_gpt2_finetuned_mistral_kd_seed_{i}.csv', 
                                seed = i, test_data_path = 'data/new_test_data_7k.csv', max_gen_length = 50)
    

# Finetuned Mistral
for i in range(5):
    generate_punchlines_mistral_corrected('models/mistral-7b-finetuned-filtered', f'./results/filtered_finetuned_mistral_seed_{i}.csv', 
                                seed = i, test_data_path = 'data/new_test_data_7k.csv', max_gen_length = 50)
    

# Zero-Shot Mistral
for i in range(5):
    generate_punchlines_zero_shot_mistral_instruct("mistralai/Mistral-7B-Instruct-v0.1", f'./results/filtered_zero_shot_mistral_seed_{i}.csv',
                                                  seed = i, test_data_path = 'data/new_test_data_7k.csv', token='', max_gen_length = 50)
    
# Gpt2-Qwen
for i in range(5):
    generate_punchlines_batched('1039351/gpt2_qwen_generator_epoch_10', f'./results/filtered_gpt2_qwen_seed_{i}.csv', 
                                seed = i, test_data_path = 'data/new_test_data_7k.csv', max_gen_length = 50)
    
# Gpt2-frozen-Qwen
for i in range(5):
    generate_punchlines_batched('1039351/gpt2_frozen_qwen_generator_epoch_10', f'./results/filtered_gpt2_frozen_qwen_seed_{i}.csv', 
                                seed = i, test_data_path = 'data/new_test_data_7k.csv',token='',max_gen_length = 50)