# Corrective-Feedback-Driven-Humor-Generation-for-Tiny-Language-Models

Models
FT GPT-2

LoRA Mistral

LoRA Qwen

LoRA LLaMA

ZS Mistral

ZS LLaMA

ZS Qwen

KD-Critique GPT-2 (Mistral)

KD-Critique GPT-2 (Qwen)

KD-Critique GPT-2 (LLaMA)

KD-Imitate ZS GPT-2 (Mistral)

KD-Imitate ZS GPT-2 (Qwen)

KD-Imitate ZS GPT-2 (LLaMA)

KD-Imitate LoRA GPT-2 (Mistral)

KD-Imitate LoRA GPT-2 (Qwen)

KD-Imitate LoRA GPT-2 (LLaMA)

G-KD-Frozen GPT-2 (Mistral)

G-KD-Frozen GPT-2 (Qwen)

G-KD-Frozen GPT-2 (LLaMA)

G-KD-LoRA GPT-2 (Mistral)

G-KD-LoRA GPT-2 (Qwen)

G-KD-LoRA GPT-2 (LLaMA)

Code
All models are implemented across the scripts/ and notebooks/ directories.
Evaluation code is available in the evaluation/ directory, and datasets are provided in the data/ directory.

gpt2_frozen_mistral.py: Contains code for GPT-2 + Frozen Mistral

gpt2_lora_mistral.py: Contains code for GPT-2 + LoRA Mistral

gpt2_kd.ipynb: Contains code for the KD baselines (KD-Imitate ZS and KD-Critique)

finetune_gpt2.py: Contains code for the FT GPT-2 baseline
