# Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations (CoD)

[**Faisal Hamman**](https://www.faisalhamman.com/), **Pasan Dissanayake**, **Yanjun Fu**, **Sanghamitra Dutta**  
University of Maryland, College Park  
Contact: {fhamman, pasand, yanjunfu, sanghamd}@umd.edu

📘 Published at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025).

---

### Overview

CoD is a few-shot distillation framework that systematically infuses counterfactual explanations (CFEs) into training to improve data efficiency and boundary fidelity. CFEs are minimally perturbed inputs that flip a teacher’s prediction, providing informative examples near decision boundaries.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
wandb login
huggingface-cli login
```
---

## Data & CFE Generation

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
3) Distill into a student using clean or CFE-infused data.  
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
bash scripts_cfx/run.sh   
```
 Multi-dataset, multi-seed orchestration examples are in `scripts_cfx/run_seed.sh`,  `scripts_cfx/run_seed_ted.sh`, and `scripts_cfx/run_seed_qwen.sh` .

---

## Tasks, Models, and Outputs

- Tasks/Datasets: `sst2`, `cola`, `imdb`, `sentiment140`, `amazon`, `yelp`.
- Teachers: examples include `microsoft/deberta-v3-base`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-7B` (see `teacher_models/`).
- Students: examples include `microsoft/deberta-v3-small`, `microsoft/deberta-v3-xsmall`, `Qwen/Qwen2.5-0.5B`.
- Outputs: training artifacts and checkpoints under `ted_output/<dataset>/*` (e.g., `kd`, `lwd`); WANDB logs under `wandb/`.


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
