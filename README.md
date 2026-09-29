# Corrective Feedback-Driven Humor Generation for Tiny Language Models

**Paper Accepted to AACL-IJCNLP 2026** (Asia-Pacific Chapter of the Association for Computational Linguistics & International Joint Conference on Natural Language Processing)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/) [![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers)

**G-KD** (**G**uided **K**nowledge **D**istillation) is a framework for teaching a *tiny* language model to be funny. A compact **GPT-2 generator (~140M parameters)** writes punchlines for joke setups, and a large LLM (**Mistral-7B**, **LLaMA-3-8B**, or **Qwen3-8B**) acts as a critic that returns **corrective feedback**. Gradients from the critic flow back into the generator through **Gumbel-Softmax** relaxation, so the small model learns directly from the large model's judgement rather than only imitating its outputs. The critic can be kept **frozen** (G-KD-Frozen) or adapted with **LoRA** (G-KD-LoRA).

---

## 📂 Repository Structure

Top-level scripts contain the reference (Mistral) implementations of the main methods. The `scripts/` directory holds the equivalent training and generation code for every critic LLM, `notebooks/` holds the knowledge-distillation baselines and teacher generation, and `evaluation/` holds all metric and LLM-as-judge code.

### Directory Layout

```
.
├── data/                                   # Datasets (setup, punchline, source, final_score)
│   ├── train_set_20k.csv                   # Training set (20k jokes)
│   ├── validation_set_5k.csv               # Validation set (5k jokes)
│   └── test_data2.csv                      # Test set
│
├── finetune_gpt2.py                        # (1) FT GPT-2 baseline (supervised fine-tuning)
├── gpt2_frozen_mistral.py                  # (2) G-KD-Frozen: GPT-2 + frozen Mistral-7B critic
├── gpt2_lora_mistral.py                    # (3) G-KD-LoRA:   GPT-2 + LoRA Mistral-7B critic
├── gpt2_kd.ipynb                           # (4) KD baselines (KD-Imitate ZS, KD-Critique)
│
├── scripts/
│   ├── gpt2_{mistral,llama,qwen}_frozen.py # G-KD-Frozen for each critic LLM
│   ├── gpt2_{mistral,llama,qween}.py       # G-KD-LoRA for each critic LLM
│   ├── gpt2_{mistral,llama,qwen}_kd.py     # KD-Critique GPT-2 for each teacher
│   ├── finetuned_{llama,qwen}.py           # LoRA fine-tuning of LLaMA / Qwen
│   ├── qwen_zero_shot.py                   # Zero-shot Qwen baseline
│   ├── generate_jokes.py                   # Shared generation utilities (GPT-2 / Mistral / ZS)
│   ├── generate_jokes_{mistral,llama,qwen}.py         # Generation from LoRA models
│   ├── generate_jokes_{llama,qwen}_zero_shot.py       # Zero-shot generation
│   ├── generate_jokes_mistral_{finetuned,gpt2}.py     # Multi-seed generation runs
│   ├── generated_jokes-filtered_run.py     # Filtered generation run
│   └── offensive_classifier.py             # Offensiveness filtering of outputs
│
├── notebooks/
│   ├── acl_generate_{mistral,llama,qwen}_sp.ipynb     # Zero-shot teacher generation
│   ├── generate_finetuned_mistral.ipynb               # LoRA Mistral generation
│   ├── gpt2_{,llama_,qwen_}imitation_phase.ipynb      # KD-Imitate (imitation phase)
│   └── gpt2_finetuned_{mistral,llama,qwen}_kd.ipynb   # KD-Imitate LoRA
│
└── evaluation/
    ├── train_bert_classifier-filtered.ipynb   # Train BERT humor classifier
    ├── get_humor_accuracy_diff_seeds.ipynb    # Humor accuracy across seeds
    ├── calculate_metrics_16july_v2-Copy1.ipynb# Automatic metrics
    ├── calculate_perplexity.py                # Fluency (perplexity)
    └── llm_judge3.py                          # Pairwise LLM-as-judge evaluation
```

### Models

| Family | Variants |
|---|---|
| **Baselines** | FT GPT-2 · LoRA Mistral / Qwen / LLaMA · ZS Mistral / Qwen / LLaMA |
| **KD-Critique** | GPT-2 (Mistral) · GPT-2 (Qwen) · GPT-2 (LLaMA) |
| **KD-Imitate ZS** | GPT-2 (Mistral) · GPT-2 (Qwen) · GPT-2 (LLaMA) |
| **KD-Imitate LoRA** | GPT-2 (Mistral) · GPT-2 (Qwen) · GPT-2 (LLaMA) |
| **G-KD-Frozen** (ours) | GPT-2 (Mistral) · GPT-2 (Qwen) · GPT-2 (LLaMA) |
| **G-KD-LoRA** (ours) | GPT-2 (Mistral) · GPT-2 (Qwen) · GPT-2 (LLaMA) |

---

## ⚙️ Installation

```bash
git clone https://github.com/Ramakrishna176420/Corrective-Feedback-Driven-Humor-Generation-for-Tiny-Language-Models.git
cd Corrective-Feedback-Driven-Humor-Generation-for-Tiny-Language-Models

pip install torch transformers peft datasets evaluate pandas numpy tqdm tabulate openai
```

LLaMA-3 and some Mistral checkpoints are gated on Hugging Face, so export a token first:

```bash
export HF_TOKEN=your_huggingface_token
```

---

## 🚀 Usage

All scripts are run from the repository root so that the `data/` paths resolve.

### 1. Train the FT GPT-2 Baseline

```bash
python finetune_gpt2.py
```

### 2. Train G-KD

**a) G-KD-Frozen** (GPT-2 generator, frozen critic)

```bash
python gpt2_frozen_mistral.py            # Mistral-7B critic
python scripts/gpt2_llama_frozen.py      # LLaMA-3-8B critic
python scripts/gpt2_qwen_frozen.py       # Qwen3-8B critic
```

**b) G-KD-LoRA** (GPT-2 generator, LoRA-adapted critic)

```bash
python gpt2_lora_mistral.py              # Mistral-7B critic
python scripts/gpt2_llama.py             # LLaMA-3-8B critic
python scripts/gpt2_qween.py             # Qwen3-8B critic
```

> **Note:** `gpt2_lora_mistral.py` loads Mistral from a local sharded copy (`../Mistral-7B-v0.1-colab-sharded`). Change `model_name` to `mistralai/Mistral-7B-v0.1` to download it from Hugging Face instead.

### 3. Train the KD Baselines

- **KD-Imitate ZS / KD-Critique (Mistral):** run `gpt2_kd.ipynb`
- **KD-Critique for each teacher:** `python scripts/gpt2_{mistral,llama,qwen}_kd.py`
- **KD-Imitate (imitation phase):** `notebooks/gpt2_*imitation_phase.ipynb`

### 4. Generate Punchlines

```bash
python scripts/generate_jokes_mistral.py          # LoRA Mistral
python scripts/generate_jokes_llama_zero_shot.py  # Zero-shot LLaMA
python scripts/generate_jokes_qwen_zero_shot.py   # Zero-shot Qwen
```

Generated outputs are written to `./results/`.

---

## 📊 Evaluation

**Humor accuracy** — train the BERT humor classifier, then score generations across seeds:

```
evaluation/train_bert_classifier-filtered.ipynb
evaluation/get_humor_accuracy_diff_seeds.ipynb
```

**Fluency (perplexity):**

```bash
python evaluation/calculate_perplexity.py
```

**Pairwise LLM-as-judge** (requires an OpenAI API key; supports sharding across jobs):

```bash
export OPENAI_API_KEY=your_openai_key
python evaluation/llm_judge3.py --num-jobs 4 --job-id 1
```

---

## 📧 Contact

For questions or inquiries, please contact:

- **Ramakrishna Pinninti**: <ramakrishna.pinninti@adaptcentre.ie>

---

## Citation

If you find this code useful, please cite:

```bibtex
@inproceedings{pinninti2026corrective,
    title     = {Corrective Feedback-Driven Humor Generation for Tiny Language Models},
    author    = {Pinninti, Ramakrishna and others},
    booktitle = {Proceedings of the Asia-Pacific Chapter of the Association for Computational Linguistics and the International Joint Conference on Natural Language Processing (AACL-IJCNLP)},
    year      = {2026},
    publisher = {Association for Computational Linguistics}
}
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
