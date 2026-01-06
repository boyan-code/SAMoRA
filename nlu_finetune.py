import ast
import os
import random
import sys
import re
from functools import partial
from typing import List

import numpy as np
import torch
import transformers
from datasets import load_dataset, load_from_disk
from deepspeed.utils.logging import LoggerFactory
from src.custom_model import LlamaForCausalLM, Qwen3ForCausalLM, Qwen3ForMultiTaskSequenceClassification, LlamaForMultiTaskSequenceClassification
from src.utils import add_filehandler, save_pretrain, set_no_grad, wrap_model
from src.utils.peft_loading_utilts import get_lora_param_maybe_zero_3, maybe_zero_3
from src.utils.dist import get_global_rank
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, TrainerCallback

logger = LoggerFactory.create_logger(__name__)

num_labels_list = [2, 2, 2, 2, 3, 2, 2]
# [cola, sst2, mrpc, qqp, mnli, qnli, rte]

TASKS = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte"]
TASK2ID = {name: i for i, name in enumerate(TASKS)}
ID2TASK = {v: k for k, v in TASK2ID.items()}

PROMPT_TEMPLATES = {
    "cola": lambda ex: f'Is the following sentence "{ex["sentence"]}" grammatically acceptable? Answer:',
    "sst2": lambda ex: f'Is the following sentence "{ex["sentence"]}" sentimently positive? Answer:',
    "mrpc": lambda ex: f'Dose the following sentence "{ex["sentence1"]}" convey the equivalent meaning as "{ex["sentence2"]}"? Answer:',
    "qqp":  lambda ex: f'Is the following question "{ex["question1"]}" essentially asking the same thing as "{ex["question2"]}"? Answer:',
    "mnli": lambda ex: f'Dose the statement "{ex["premise"]}" imply that "{ex["hypothesis"]}" ? Answer:',
    "qnli": lambda ex: f'Based on the statement: "{ex["question"]}" dose the following sentence "{ex["sentence"]}" have a definitive answer? Answer:',
    "rte":  lambda ex: f'Dose the text "{ex["sentence1"]}" entail the statement "{ex["sentence2"]}"? Answer:',
}


def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}"""


import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="gpt2",
        help="base model to use for finetuning",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="cache tokenized data",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="yahma/alpaca-cleaned",
        help="path to dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./homelora-alpaca",
        help="output directory",
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default="homelora",
        help="adapter type to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="batch size",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="number of epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="learning rate scheduler type",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.01,
        help="warmup ratio",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="weight decay",
    )
    parser.add_argument(
        "--cutoff_len",
        type=int,
        default=256,
        help="max sequence length",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        help="use gradient checkpointing",
    )
    parser.add_argument(
        "--save_step",
        type=int,
        default=200,
        help="save step",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="lora r",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="lora alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="lora dropout",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=ast.literal_eval,
        default=None,
        help="lora target modules",
    )
    # mlora hyperparams
    parser.add_argument(
        "--lambda_num",
        type=int,
        default=3,
        help="lambda num",
    )
    parser.add_argument(
        "--num_B",
        type=int,
        default=3,
        help="num B",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature (B_scale for samora)",
    )
    parser.add_argument(
        "--diagonal_format",
        type=bool,
        default=True,
        help="diagonal format for lambda matrices",
    )
    parser.add_argument(
        "--tunable_scaler",
        type=bool,
        default=False,
        help="use tunable scaler",
    )
    # multilora hyperparams
    parser.add_argument(
        "--lora_num",
        type=int,
        default=3,
        help="lora num",
    )
    # moelora hyperparams
    parser.add_argument(
        "--expert_num",
        type=int,
        default=3,
        help="expert num",
    )
    parser.add_argument(
        "--task_num",
        type=int,
        default=8,
        help="task num",
    )
    parser.add_argument(
        "--te_dim",
        type=int,
        default=64,
        help="te dim",
    )
    # dora hyperparams
    parser.add_argument(
        "--merge_weights",
        type=bool,
        default=False,
        help="merge weights",
    )
    parser.add_argument(
        "--Wdecompose",
        type=bool,
        default=False,
        help="Wdecompose",
    )
    parser.add_argument(
        "--dora_simple",
        type=bool,
        default=True,
        help="dora simple",
    )
    # misc
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient accumulation steps",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="wandb project",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="wandb run name",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--load_from_checkpoint",
        type=str,
        default=None,
        help="load from checkpoint",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default="",
        help="deepspeed",
    )
    parser.add_argument(
        "--train_on_inputs",
        type=bool,
        default=False,
        help="train on inputs",
    )
    parser.add_argument(
        "--use_svd_init",
        action="store_true",
        help="use SVD results to initialize lora_A and lora_B",
    )
    parser.add_argument(
        "--svd_path",
        type=str,
        default="your_svd_path",
        help="path to SVD results directory",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
    )

    return parser.parse_args()


def _init_from_svd(model, svd_path: str, lora_r: int, num_B: int, logger):
    """Initialize lora_A and lora_B from SVD results."""
    if not os.path.exists(svd_path):
        logger.warning(f"SVD path {svd_path} does not exist, skipping SVD initialization")
        return
    
    # Module name mapping: model layer name -> SVD file name pattern
    module_name_mapping = {
        "q_proj": "self_attn_q_proj",
        "k_proj": "self_attn_k_proj",
        "v_proj": "self_attn_v_proj",
        "o_proj": "self_attn_o_proj",
        "gate_proj": "mlp_gate_proj",
        "up_proj": "mlp_up_proj",
        "down_proj": "mlp_down_proj",
    }
    
    initialized_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            match = re.search(r'layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)', name)
            if match:
                layer_idx = int(match.group(1))
                module_name = match.group(3)  # q_proj, k_proj, etc.
                
                if module_name in module_name_mapping:
                    svd_module_name = module_name_mapping[module_name]
                    svd_file = os.path.join(
                        svd_path, 
                        f"model_layers_{layer_idx}_{svd_module_name}_svd.pt"
                    )
                    
                    if not os.path.exists(svd_file):
                        logger.warning(f"SVD file not found for {name}: {svd_file}")
                        continue
                    
                    try:
                        svd_data = torch.load(svd_file, map_location="cpu")
                        U = svd_data["U"] 
                        S = svd_data["S"]
                        Vh = svd_data["Vh"]
                        
                        
                        in_features = module.lora_A.shape[-1]
                        out_features = module.lora_B.shape[1]
                        S_diag = S[:lora_r].to(module.lora_B.device).to(module.lora_B.dtype)
                        
                        with torch.no_grad():
                            # module.lora_A.data[0] = lora_A_init
                            module.lora_scale.data = S_diag
                            # for b_idx in range(num_B):
                            #     module.lora_B.data[b_idx] = lora_B_init
                        
                        initialized_count += 1
                        logger.info(f"Initialized {name} from SVD")
                    except Exception as e:
                        logger.warning(f"Failed to initialize {name} from SVD: {e}")
                        continue
    
    logger.info(f"Initialized {initialized_count} modules from SVD results")


def train(
    base_model: str = "",
    cache_dir: str = None,  # cache tokenized data
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./homelora-alpaca",
    adapter_name: str = "homelora",
    batch_size: int = 128,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.01,
    weight_decay: float = 0.0,
    cutoff_len: int = 256,
    use_gradient_checkpointing: bool = False,
    save_step: int = 200,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    # mlora hyperparams
    lambda_num: int = 3,
    num_B: int = 3,
    temperature: float = 1.0,
    diagonal_format: bool = True,
    tunable_scaler: bool = False,
    # multilora hyperparams
    lora_num: int = 3,
    # moelora hyperparams
    expert_num: int = 3,
    task_num: int = 8,
    te_dim: int = 64,
    # misc
    gradient_accumulation_steps: int = 1,
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter (resume training)
    load_from_checkpoint: str = None,  # either training checkpoint or final adapter
    deepspeed: str = "",
    train_on_inputs: bool = False,
    merge_weights: bool = False,
    Wdecompose: bool = False,
    dora_simple: bool = True,
    use_svd_init: bool = False,
    svd_path: str = "your_svd_path",
    **kwargs,
):
    # Set random seed for reproducibility
    seed = 41
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")
    
    use_wandb = wandb_project is not None
    add_filehandler(logger, os.path.join(output_dir, "logging"))
    
    # Load model config first to check model type
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    
    # For sequence classification, use multi-task classification model
    if config.model_type == "qwen3":
        # Load base model first, then wrap with classification head
        base_model_for_classification = Qwen3ForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        # Create multi-task classification model
        model = Qwen3ForMultiTaskSequenceClassification(
            config,
            num_labels_list=num_labels_list
        )
        # Copy the base model weights
        model.model = base_model_for_classification.model
    elif config.model_type == "llama":
        # Load base model first, then wrap with classification head
        base_model_for_classification = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        # Create multi-task classification model
        model = LlamaForMultiTaskSequenceClassification(
            config,
            num_labels_list=num_labels_list
        )
        # Copy the base model weights
        model.model = base_model_for_classification.model
    else:
        raise ValueError(f"Multi-task sequence classification currently only supports qwen3 and llama, got {config.model_type}")

    # Load tokenizer
    if config.model_type == "llama":
        # For Llama-3 models, use AutoTokenizer; for Llama-2 and earlier, use LlamaTokenizer
        if "Llama-3" in base_model or "llama-3" in base_model.lower():
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = "right"
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.enable_input_require_grads()

    if adapter_name.lower() == "mlora":
        mlora_config = {
            "type": "mlora",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lambda_num": lambda_num,
            "B_num": num_B,
            "B_scale": temperature,
            "diagonal_format": False,
        }
    elif adapter_name.lower() == "multilora":
        mlora_config = {
            "type": "multilora",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_num": lora_num,
        }
    elif adapter_name.lower() == "moelora":
        mlora_config = {
            "type": "moelora",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "expert_num": expert_num,
            "task_num": task_num,
            "task_embedding_dim": te_dim,
        }
    elif adapter_name.lower() == "hydralora":
        mlora_config = {
            "type": "hydralora",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_num": lora_num,
            "B_scale": temperature,
        }
    elif adapter_name.lower() == "dora":
        mlora_config = {
            "type": "dora",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "merge_weights": merge_weights,
            "Wdecompose": Wdecompose,
            "dora_simple": dora_simple,
        }
    elif adapter_name.lower() in ["samora", "laser", "homelora", "samora"]:
        mlora_config = {
            "type": "samora",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lambda_num": lambda_num,
            "B_num": num_B,
            "B_scale": temperature,
            "diagonal_format": diagonal_format,
            "tunable_scaler": tunable_scaler,
        }
    elif adapter_name.lower() == "lora":
        mlora_config = {
            "type": "lora",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "merge_weights": merge_weights,
        }
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_name}")

    model = wrap_model(model, lora_target_modules, mlora_config)
    
    # Initialize lora_A and lora_B from SVD results if enabled
    if use_svd_init and adapter_name.lower() in ["samora", "laser", "homelora", "samora"]:
        logger.info(f"Initializing lora_A and lora_B from SVD results in {svd_path}")
        _init_from_svd(model, svd_path, lora_r, num_B, logger)
    
    if load_from_checkpoint is not None:
        state_dict = torch.load(load_from_checkpoint, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(msg.unexpected_keys)
    set_no_grad(model, logger=logger)
    
    # Enable training for classification head (score)
    for name, param in model.named_parameters():
        if "score" in name or "heads" in name:
            param.requires_grad = True
            logger.info(f"Enabled training for classification head: {name}")
    
    model.config.use_cache = False

    def tokenize_per_example(data_point):
        """Tokenize a single example for sequence classification."""
        tid = data_point.get("task_id", 0)
        task_name = ID2TASK.get(tid, TASKS[0])
        
        if task_name in PROMPT_TEMPLATES:
            prompt = PROMPT_TEMPLATES[task_name](data_point)
        else:
            # Fallback to generic prompt
            prompt = generate_prompt(data_point)
        
        tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        
        tokens["labels"] = int(data_point.get("label", 0))
        tokens["task_id"] = tid
        
        if adapter_name.lower() in ["mlora", "moelora", "hydralora", "samora", "laser", "homelora", "samora"]:
            tokens["lambda_index"] = tid
        
        return tokens

    if cache_dir is not None and os.path.exists(os.path.join(cache_dir, "train")):
        train_data = load_from_disk(os.path.join(cache_dir, "train"))
    else:
        if data_path.endswith(".json"):  # todo: support jsonl
            data = load_dataset("json", data_files=data_path)
        else:
            data = load_dataset(data_path)

        train_data = (
            data["train"]
            .shuffle()
            .map(
                tokenize_per_example,
                batched=False,
                desc="Tokenizing train data (multi-task classification)",
                num_proc=4,
            )
        )

        if cache_dir is not None:
            train_data.save_to_disk(os.path.join(cache_dir, "train"))

    # Custom data collator to handle lambda_index, task_id, and classification labels
    class MultiTaskClassificationDataCollator:
        def __init__(self, tokenizer, padding=True, max_length=None):
            self.tokenizer = tokenizer
            self.padding = padding
            self.max_length = max_length
        
        def __call__(self, features, return_tensors="pt"):
            # Extract special fields
            lambda_indices = None
            task_ids = None
            labels = None
            
            if "lambda_index" in features[0]:
                lambda_indices = [f.pop("lambda_index") for f in features]
            if "task_id" in features[0]:
                task_ids = [f.pop("task_id") for f in features]
            # Handle both "label" (singular) and "labels" (plural)
            if "labels" in features[0]:
                labels = [f.pop("labels") for f in features]
            elif "label" in features[0]:
                labels = [f.pop("label") for f in features]
            
            # Tokenize and pad
            batch = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                return_tensors=return_tensors,
            )
            
            # Add special fields back
            if lambda_indices is not None:
                lambda_indices = [min(max(int(idx), 0), lambda_num - 1) for idx in lambda_indices]
                batch["lambda_index"] = torch.tensor(lambda_indices, dtype=torch.long)
            
            if task_ids is not None:
                batch["task_ids"] = torch.tensor(task_ids, dtype=torch.long)
            
            if labels is not None:
                batch["labels"] = torch.tensor(labels, dtype=torch.long)
            
            return batch

    # Custom Trainer to pass task_ids and lambda_index to model forward
    class MultiTaskTrainer(transformers.Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """
            Override compute_loss to pass task_ids and lambda_index to model forward.
            
            Args:
                num_items_in_batch: Optional parameter passed by transformers Trainer (can be ignored)
            """
            task_ids = inputs.pop("task_ids", None)
            # Handle both "label" (singular) and "labels" (plural) from dataset
            labels = inputs.pop("label", None)
            lambda_index = inputs.pop("lambda_index", None)
            
            outputs = model(**inputs, task_ids=task_ids, lambda_index=lambda_index)
            loss = outputs.loss if hasattr(outputs, "loss") else None
            
            return (loss, outputs) if return_outputs else loss

    trainer = MultiTaskTrainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            weight_decay=weight_decay,
            bf16=True,
            deepspeed=deepspeed,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="no",
            save_strategy="no",
            save_only_model=True,
            eval_steps=None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            report_to=(["wandb", "tensorboard"] if use_wandb else ["tensorboard"]),
            run_name=wandb_run_name if use_wandb else None,
            seed=41,
        ),
        data_collator=MultiTaskClassificationDataCollator(
            tokenizer, padding=True, max_length=cutoff_len
        ),
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save LoRA weights
    save_pretrain(model, output_dir, prefix=["lora"])
    
    # Save classification head separately
    if get_global_rank() == 0:
        classification_head_state_dict = {}
        for name, param in model.named_parameters():
            if "score" in name or "heads" in name:
                classification_head_state_dict[name] = maybe_zero_3(param, ignore_status=True)
        
        if classification_head_state_dict:
            output_dir_checkpoint = os.path.join(output_dir, "checkpoint")
            os.makedirs(output_dir_checkpoint, exist_ok=True)
            heads_path = os.path.join(output_dir_checkpoint, "heads.pt")
            torch.save(classification_head_state_dict, heads_path)
            logger.info(f"Classification head saved to {heads_path}")


if __name__ == "__main__":
    args = arg_parser()
    train(
        **vars(args),
    )
