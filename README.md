# Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations (CoD)

**Faisal Hamman**, **Pasan Dissanayake**, **Yanjun Fu**, **Sanghamitra Dutta**  
University of Maryland, College Park  
Contact: {fhamman, pasand, yanjunfu, sanghamd}@umd.edu

📘 Published at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025).

---

## Overview

CoD is a few-shot distillation framework that systematically infuses counterfactual explanations (CFEs) into training to improve data efficiency and boundary fidelity. CFEs are minimally perturbed inputs that flip a teacher’s prediction, providing informative examples near decision boundaries.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
# Optional: Weights & Biases for experiment tracking
wandb login
```

> Note: Some teacher/student models (e.g., `Qwen` and `microsoft` families) may require Hugging Face authentication or model access. Ensure your `huggingface-cli login` is configured where needed.

---

## Data & Counterfactual Generation

Data preparation and CFEs are handled in `cfx-generator/`. Outputs are saved under:
- Clean/base data: `cfx-generator/dataset/<dataset>_<subset>_seed<seed>`
- CFEs (concatenated): `cfx-generator/cfx-dataset/<dataset>_<subset>_seed<seed>`

Supported datasets: `sst2`, `cola`, `imdb`, `sentiment140`, `amazon`, `yelp`.

### 1) Generate clean/base subsets

Run the batch script (multiple datasets, sizes, and seeds):

```bash
bash cfx-generator/gen_data.sh
```

### 2) Generate CFEs with an LLM

Set your OpenAI API key and run the batch script:

```bash
# Replace with your key
export OPENAI_API_KEY="API_KEY"

bash cfx-generator/gen_cfx.sh
```

---

## Training & Distillation (CoD)

Training and distillation scripts live in `scripts_cfx/`. A typical workflow is:

1) Generate clean data and CFEs (above).  
2) Train or load a teacher.  
3) Distill into a student using clean or CFE-augmented data.  
4) Track metrics via WANDB; artifacts saved in `ted_output/`.

### Train Teachers

Train across multiple datasets:

```bash
bash scripts_cfx/train-teacher.sh
bash scripts_cfx/train-teacher-qwen.sh
```

Teachers are saved under `teacher_models/<MODEL>/<dataset>/teacher_init`.

### Distillation (KD / Layer-wise Distillation)

Batch  runs with and without CFEs:

```bash        
bash scripts_cfx/dist_glue.sh   
```

Example single-run (KD, clean):

```bash
python text-classification/ted_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-small \
  --teacher_model_name_or_path cliang1453/deberta-v3-base-sst2 \
  --model_type ted-deberta-v2 \
  --task_name sst2 \
  --per_device_train_batch_size 8 \
  --max_length 256 \
  --learning_rate 6e-5 --num_train_epochs 75 \
  --num_warmup_steps 0 --teacher_filter_interval 2 \
  --kl_alpha 20 --mse_alpha 0 --filter_disabled \
  --output_dir ted_output/sst2/kd \
  --seed 42 --mixed_precision fp16 --save_best \
  --subset_size 32 --data_type clean --with_tracking
```

Use CFEs by setting `--data_type cfx`:

```bash
python text-classification/ted_no_trainer.py \
  ... \
  --subset_size 32 --data_type cfx --with_tracking
```

> Notes: Adjust `--task_name`, `--subset_size`, and paths to match generated data. Multi-dataset, multi-seed orchestration examples are in `scripts_cfx/run.sh` and `scripts_cfx/run_seed.sh`.

---

## Tasks, Models, and Outputs

- Tasks/Datasets: `sst2`, `cola`, `imdb`, `sentiment140`, `amazon`, `yelp`.
- Teachers: examples include `microsoft/deberta-v3-base`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-7B` (see `teacher_models/`).
- Students: examples include `microsoft/deberta-v3-small`, `microsoft/deberta-v3-xsmall`, `Qwen/Qwen2.5-0.5B`.
- Outputs: training artifacts and checkpoints under `ted_output/<dataset>/*` (e.g., `kd`, `lwd`); WANDB logs under `wandb/`.

---

## Reproducibility & Logging

- Seeds: see `scripts_cfx/run_seed.sh`, `scripts_cfx/run_seed_ted.sh`, `scripts_cfx/run_seed_qwen.sh` for multi-seed orchestration.
- Tracking: pass `--with_tracking` to log to WANDB; results are saved locally under `ted_output/`.

---

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@inproceedings{hamman2025fewshotcod,
  title     = {Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations},
  author    = {Hamman, Faisal and Dissanayake, Pasan and Fu, Yanjun and Dutta, Sanghamitra},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

This implementation builds upon TED:

```bibtex
@article{liang2022less,
  title={Less is More: Task-aware Layer-wise Distillation for Language Model Compression},
  author={Liang, Chen and Zuo, Simiao and Zhang, Qingru and He, Pengcheng and Chen, Weizhu and Zhao, Tuo},
  journal={arXiv preprint arXiv:2210.01351},
  year={2022}
}
```

---
