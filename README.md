# SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning

Official implementation of **SAMoRA** ([Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1404/)), a parameter-efficient fine-tuning method that combines a shared low-rank projection with a semantic-aware router over multiple LoRA "B" experts and a task-adaptive gate.

## Method at a glance

Each adapted linear layer (`src/adapter/samora.py`, class `SALinear`) consists of:

- **Shared down-projection `lora_A`** (`r × in_features`) with a learnable per-rank scale `lora_scale`.
- **`B_num` expert up-projections `lora_B`** (`B_num × out_features × r`).
- **Semantic-aware router**: tokens are routed by the cosine similarity between the low-rank activation and per-expert keys `lora_lambdas`, softmax-normalized with temperature `--temperature`.
- **Task-adaptive gate**: a per-task embedding (`--lambda_num` tasks) passed through a sigmoid gate scales the expert output. For unseen tasks (`lambda_index = -1`), the average gate over all trained tasks is used.
- **Regularization** (training only): orthogonality losses on `lora_A`/`lora_B` plus a KL alignment term between router keys and expert representations, added to the LM loss automatically inside the custom model.

## Repository structure

```
├── samora_finetune.py      # training on instruction / commonsense-reasoning data (causal LM)
├── samora_evaluate.py      # evaluation on commonsense benchmarks (boolq, piqa, siqa, hellaswag, winogrande, ARC, obqa, csqa)
├── nlu_finetune.py         # multi-task GLUE training (sequence classification heads)
├── nlu_evaluate.py         # GLUE evaluation
├── config/ds2.json         # DeepSpeed ZeRO-2 config
├── script/                 # example train / eval shell scripts (Llama & Qwen3)
└── src/
    ├── adapter/            # SALinear (SAMoRA), LoRALinear, DoRALinear
    ├── custom_model/       # Llama / Qwen3 modeling files that thread lambda_index & adapter losses
    └── utils/              # model wrapping, checkpoint saving (ZeRO-3 aware), logging
```

## Installation

```bash
conda create -n samora python=3.10 -y
conda activate samora
pip install -r requirements.txt
```

Qwen3 support requires `transformers >= 4.51`.

## Data format

Training data is a JSON file of Alpaca-style records; `task_id` selects the task embedding (defaults to 0 if absent):

```json
{"instruction": "...", "input": "...", "output": "...", "task_id": 2}
```

For commonsense reasoning we follow the multi-task setup of LLM-Adapters (8 tasks + csqa; see `task_name_to_id` in `samora_evaluate.py`). For NLU we use 7 GLUE tasks (`cola, sst2, mrpc, qqp, mnli, qnli, rte`; see `TASKS` in `nlu_finetune.py`).

## Training

Edit the placeholder paths (`your_*`) in the scripts under `script/`, then:

```bash
# commonsense reasoning, Llama
bash script/llama2_7B_samora_qkvo.sh

# commonsense reasoning, Qwen3-8B
bash script/qwen3_8B_samora_qkvo_train.sh

# GLUE multi-task classification
bash script/llama3_8b_nlu_train.sh     # or qwen3_8b_nlu_train.sh
```

Key SAMoRA hyperparameters:

| Flag | Meaning | Typical |
|---|---|---|
| `--lora_r` / `--lora_alpha` | LoRA rank / scaling | 8 / 16 |
| `--num_B` | number of B experts | 3 |
| `--lambda_num` | number of tasks (task-gate embeddings) | #tasks (+1 spare) |
| `--temperature` | router softmax temperature | 0.8 |
| `--lora_target_modules` | wrapped layers | `'["q_proj","k_proj","v_proj","o_proj"]'` |
| `--use_svd_init` / `--svd_path` | initialize `lora_scale` from pre-computed SVD of base weights | optional |

Boolean flags (`--diagonal_format`, `--tunable_scaler`, …) accept `True/False` strings.

Training saves only the adapter parameters to `<output_dir>/checkpoint/final_checkpoint.pt` (NLU additionally saves classification heads to `heads.pt`).

## Evaluation

```bash
# commonsense reasoning
DATASET=boolq bash script/qwen3_8b_samora_qkvo_eval.sh
# or directly:
CUDA_VISIBLE_DEVICES=0 python samora_evaluate.py \
    --model Qwen3 --adapter samora --dataset boolq \
    --base_model <base_model_path> \
    --lora_weights <output_dir>/checkpoint/final_checkpoint.pt \
    --lora_target_modules '["q_proj","k_proj","v_proj","o_proj"]' \
    --batch_size 32 --lora_r 8 --lora_alpha 16 \
    --lambda_num 9 --num_B 3 --temperature 0.8

# GLUE
TASK=rte bash script/qwen3_8b_nlu_eval.sh
```

Set the dataset root inside `samora_evaluate.py` / `nlu_evaluate.py` (`your_dataset_root`, `your_glue_root`) to where your test JSON files live. Predictions and accuracy are written to `experiment/`.

Evaluation must be run with the **same adapter hyperparameters** (`--lora_r`, `--num_B`, `--lambda_num`, `--temperature`, target modules) used at training time, otherwise the checkpoint will not load.

## Baselines

`--adapter_name lora` and `--adapter_name dora` train the LoRA / DoRA baselines with the same pipeline. (`laser` and `homelora` are accepted as legacy aliases of `samora`.)

## Citation

```bibtex
@inproceedings{shi-etal-2026-samora,
    title = "{SAM}o{RA}: Semantic-Aware Mixture of {L}o{RA} Experts for Task-Adaptive Learning",
    author = "Shi, Boyan  and
      Chen, Wei  and
      Zhao, Shuyuan  and
      Shen, Junfeng  and
      Guo, Shengnan  and
      Wang, Shaojiang  and
      Wan, Huaiyu",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1404/",
    doi = "10.18653/v1/2026.findings-acl.1404",
    pages = "28173--28188",
    ISBN = "979-8-89176-395-1"
}
```
