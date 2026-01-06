import argparse
import ast
import copy
import json
import os
import re
import sys

import torch
from src.custom_model import LlamaForCausalLM, Qwen3ForCausalLM, Qwen3ForMultiTaskSequenceClassification, LlamaForMultiTaskSequenceClassification
from src.utils import wrap_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaTokenizer,
    set_seed,
)
from sklearn.metrics import matthews_corrcoef, accuracy_score
from scipy.stats import pearsonr

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# NLU tasks
NLU_TASKS = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte"]
NLU_TASK2ID = {name: i for i, name in enumerate(NLU_TASKS)}
NLU_ID2TASK = {v: k for k, v in NLU_TASK2ID.items()}

# Generation tasks
GEN_TASK_NAME_TO_ID = {
    "cola": 0,
    "sst2": 1,
    "mrpc": 2,
    "qqp": 3,
    "mnli": 4,
    "qnli": 5,
    "rte": 6,
}

# NLU prompt templates
PROMPT_TEMPLATES = {
    "cola": lambda ex: f'Is the following sentence "{ex["sentence"]}" grammatically acceptable? Answer:',
    "sst2": lambda ex: f'Is the following sentence "{ex["sentence"]}" sentimently positive? Answer:',
    "mrpc": lambda ex: f'Dose the following sentence "{ex["sentence1"]}" convey the equivalent meaning as "{ex["sentence2"]}"? Answer:',
    "qqp":  lambda ex: f'Is the following question "{ex["question1"]}" essentially asking the same thing as "{ex["question2"]}"? Answer:',
    "mnli": lambda ex: f'Dose the statement "{ex["premise"]}" imply that "{ex["hypothesis"]}" ? Answer:',
    "qnli": lambda ex: f'Based on the statement: "{ex["question"]}" dose the following sentence "{ex["sentence"]}" have a definitive answer? Answer:',
    "rte":  lambda ex: f'Dose the text "{ex["sentence1"]}" entail the statement "{ex["sentence2"]}"? Answer:',
}

# NLU num_labels_list: [cola, sst2, mrpc, qqp, mnli, qnli, rte]
NLU_NUM_LABELS = [2, 2, 2, 2, 3, 2, 2]


def main():
    args = parse_args()
    is_nlu_task = args.dataset in NLU_TASKS

    def evaluate_generation(
        instructions,
        input=None,
        temperature=1.0,
        top_p=1,
        top_k=1,
        num_beams=1,
        max_new_tokens=32,
        **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        if args.adapter in ["mlora", "moelora", "samora", "homelora", "samora"]:
            lambda_index = GEN_TASK_NAME_TO_ID[args.dataset]
            lambda_index = (
                torch.tensor(lambda_index).repeat(input_ids.shape[0]).to(device)
            )
        generation_config = GenerationConfig(
            temperature=temperature,
            # top_p=top_p,
            # top_k=top_k,
            num_beams=num_beams,
            do_sample=False,
            **kwargs,
        )
        with torch.no_grad():
            if args.adapter in ["mlora", "moelora", "samora", "homelora", "samora"]:
                generation_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    lambda_index=lambda_index,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                    use_cache=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            else:
                generation_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                    use_cache=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("### Response:")[1].strip() for o in outputs]
        return outputs

    def evaluate_nlu(batch_data):
        """Evaluate NLU tasks using classification model."""
        prompts = []
        task_ids = []
        for data in batch_data:
            task_name = args.dataset
            if task_name in PROMPT_TEMPLATES:
                prompt = PROMPT_TEMPLATES[task_name](data)
            else:
                prompt = data.get("instruction", "")
            prompts.append(prompt)
            task_ids.append(NLU_TASK2ID[task_name])
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        task_ids_tensor = torch.tensor(task_ids, dtype=torch.long).to(device)
        
        lambda_index = None
        if args.adapter in ["mlora", "moelora", "samora", "homelora", "samora"]:
            lambda_index = task_ids_tensor
        
        with torch.no_grad():
            if args.adapter in ["mlora", "moelora", "samora", "homelora", "samora"]:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_ids=task_ids_tensor,
                    lambda_index=lambda_index,
                    use_cache=False,  # Disable cache for NLU tasks
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_ids=task_ids_tensor,
                    use_cache=False,  # Disable cache for NLU tasks
                )
        
        logits = outputs.logits
        if isinstance(logits, list):
            batch_logits = torch.cat([x.unsqueeze(0) if x.dim() == 1 else x for x in logits], dim=0)
        else:
            batch_logits = logits
        
        if batch_logits.dim() == 3:
            selected_logits = []
            for i, task_id in enumerate(task_ids):
                if task_id < batch_logits.shape[1]:
                    selected_logits.append(batch_logits[i, task_id])
                else:
                    selected_logits.append(batch_logits[i, 0])
            batch_logits = torch.stack(selected_logits, dim=0)
        
        predictions = batch_logits.argmax(dim=-1).cpu().numpy()
        return predictions

    save_file = f"experiment/{args.model}-{args.adapter}-{args.dataset}.json"
    os.makedirs("experiment", exist_ok=True)

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args)

    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    all_labels = []
    all_preds = []
    pbar = tqdm(total=total)
    model.to(torch.bfloat16)
    model.eval()
    
    for idx, batch in enumerate(batches):
        current += len(batch)
        
        if is_nlu_task:
            predictions = evaluate_nlu(batch)
            
            for data, pred in zip(batch, predictions):
                label = int(data.get("label", -1))
                pred_int = int(pred)  # Convert numpy int to Python int
                all_labels.append(label)
                all_preds.append(pred_int)
                
                flag = bool(label == pred_int)  # Ensure Python bool
                if flag:
                    correct += 1
                
                new_data = copy.deepcopy(data)
                new_data["pred"] = pred_int
                new_data["label"] = label
                new_data["flag"] = flag
                output_data.append(new_data)
                
                task_name = args.dataset
                if task_name in PROMPT_TEMPLATES:
                    prompt = PROMPT_TEMPLATES[task_name](data)
                else:
                    prompt = data.get("instruction", "")
                print(prompt)
                print("prediction:", pred)
                print("label:", label)
        else:
            instructions = [data.get("instruction") for data in batch]
            outputs = evaluate_generation(instructions)

            for data, output in zip(batch, outputs):
                label = data.get("answer")
                flag = False
                predict = extract_answer(args, output)
                if label == predict or (predict in label and predict != ""):
                    correct += 1
                    flag = True
                new_data = copy.deepcopy(data)
                new_data["output_pred"] = output
                new_data["pred"] = predict
                new_data["flag"] = flag
                output_data.append(new_data)
                print(data["instruction"])
                print(output)
                print("prediction:", predict)
                print("label:", label)
        
        print("---------------")
        if is_nlu_task:
            metrics = compute_nlu_metrics(args.dataset, all_labels, all_preds)
            print(f"\rtest:{idx + 1}/{total} | accuracy {correct}/{current} = {correct / current:.4f} | {metrics}")
        else:
            print(f"\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}")
        print("---------------")
        with open(save_file, "w+") as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    
    if is_nlu_task and len(all_labels) > 0:
        final_metrics = compute_nlu_metrics(args.dataset, all_labels, all_preds)
        print("\nFinal metrics:", final_metrics)
    
    print("\n")
    print("test finished")


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def compute_nlu_metrics(task, labels, preds):
    """Compute metrics for NLU tasks."""
    if task == "cola":
        # MCC
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task == "stsb":
        # Pearson
        return {"pearson": pearsonr(labels, preds)[0]}
    else:
        # acc
        return {"acc": accuracy_score(labels, preds)}


def load_data(args) -> list:
    """read data from dataset file"""
    if args.dataset in NLU_TASKS:
        file_path = f"your_glue_root/glue_{args.dataset}/validation.json"
        if not os.path.exists(file_path):
            file_path = f"your_glue_root/glue_{args.dataset}/test.json"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"can not find dataset file : {file_path}")
        json_data = json.load(open(file_path, "r"))
        for data in json_data:
            if "task_id" not in data:
                data["task_id"] = NLU_TASK2ID[args.dataset]
        return json_data
    else:
        file_path = f"your_dataset_root/{args.dataset}/test.json"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"can not find dataset file : {file_path}")
        json_data = json.load(open(file_path, "r"))
        return json_data


def create_batch(dataset, batch_size):
    batches = []
    num_batch = (
        len(dataset) // batch_size
        if len(dataset) % batch_size == 0
        else len(dataset) // batch_size + 1
    )
    for i in range(num_batch):
        batch = dataset[i * batch_size : min((i + 1) * batch_size, len(dataset))]
        batches.append(batch)
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[
            "boolq",
            "piqa",
            "social_i_qa",
            "hellaswag",
            "winogrande",
            "ARC-Challenge",
            "ARC-Easy",
            "openbookqa",
            "cola",
            "sst2",
            "mrpc",
            "qqp",
            "mnli",
            "qnli",
            "rte",
        ],
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=["LLaMA-7B", "LLaMA-13B", "BLOOM-7B", "GPT-j-6B", "Qwen3"],
        required=True,
    )
    parser.add_argument(
        "--adapter",
        choices=["mlora", "moelora", "multilora", "dora", "samora", "homelora", "lora", "hydralora", "samora"],
        required=True,
    )
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_weights", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lora_target_modules", type=ast.literal_eval, required=True)
    parser.add_argument("--lora_r", type=int, required=True)
    parser.add_argument("--lora_alpha", type=int, required=True)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # mlora/samora hyperparams
    parser.add_argument("--lambda_num", type=int)
    parser.add_argument("--num_B", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--diagonal_format", type=bool, default=True)
    parser.add_argument("--tunable_scaler", type=bool, default=False)
    # multilora
    parser.add_argument("--lora_num", type=int)
    # moelora
    parser.add_argument("--expert_num", type=int)
    parser.add_argument("--task_num", type=int)
    parser.add_argument("--te_dim", type=int)
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
    # Optional: path to classification heads (for NLU tasks)
    parser.add_argument(
        "--heads_path",
        type=str,
        default=None,
        help="path to classification heads weights (optional, for NLU tasks). If not provided, will automatically search in {lora_weights_dir}/checkpoint/heads.pt or {lora_weights_dir}/heads.pt",
    )

    return parser.parse_args()


def print_parameter_count(state_dict, model, is_nlu_task=False):
    """Print parameter statistics."""
    def count_parameters(param_dict):
        """计算参数字典中的总参数量"""
        total = 0
        for key, value in param_dict.items():
            if isinstance(value, torch.Tensor):
                total += value.numel()
        return total
    
    def format_number(num):
        """格式化数字，显示为可读格式"""
        if num >= 1e9:
            return f"{num / 1e9:.2f}B"
        elif num >= 1e6:
            return f"{num / 1e6:.2f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.2f}K"
        else:
            return str(num)
    
    print("\n" + "="*60)
    print("Parameter Count Statistics")
    print("="*60)
    
    state_dict_count = count_parameters(state_dict)
    print(f"\n1. State Dict (LoRA weights):")
    print(f"   {state_dict_count:,} ({format_number(state_dict_count)})")
    
    model_total = sum(p.numel() for p in model.parameters())
    model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n2. Model parameter count:")
    print(f"   Total params: {model_total:,} ({format_number(model_total)})")
    print(f"   Trainable params: {model_trainable:,} ({format_number(model_trainable)})")
    print(f"   Frozen params: {model_total - model_trainable:,} ({format_number(model_total - model_trainable)})")
    
    if is_nlu_task:
        if hasattr(model, "score"):
            head_params = dict(model.score.named_parameters())
            head_count = count_parameters(head_params)
            print(f"\n3. Head parameter count:")
            print(f"   {head_count:,} ({format_number(head_count)})")
            print(f"   Head parameter details:")
            for name, param in head_params.items():
                print(f"     {name}: {param.shape} ({param.numel():,} params)")
        else:
            print(f"\n3. Head parameter count:")
            print(f"   No head found (model.score missing)")
    else:
        print(f"\n3. Head parameter count:")
        print(f"   Not an NLU task, no classification head")
    
    print("="*60 + "\n")


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f"can not find base model name by the value: {args.model}")
    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(f"can not find lora weight, the value is: {lora_weights}")

    is_nlu_task = args.dataset in NLU_TASKS

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    
    if config.model_type == "llama":
        if "Llama-3" in base_model or "LLaMA-3" in args.model or "llama-3" in base_model.lower():
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token

    # Load classification model for NLU tasks
    if is_nlu_task:
        if config.model_type == "qwen3":
            base_model_for_classification = Qwen3ForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
                trust_remote_code=True,
            )
            model = Qwen3ForMultiTaskSequenceClassification(
                config,
                num_labels_list=NLU_NUM_LABELS
            )
            # Copy the base model weights
            model.model = base_model_for_classification.model
            if tokenizer.pad_token_id is not None:
                model.config.pad_token_id = tokenizer.pad_token_id
        elif config.model_type == "llama":
            base_model_for_classification = LlamaForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
                trust_remote_code=True,
            )
            model = LlamaForMultiTaskSequenceClassification(
                config,
                num_labels_list=NLU_NUM_LABELS
            )
            # Copy the base model weights
            model.model = base_model_for_classification.model
            if tokenizer.pad_token_id is not None:
                model.config.pad_token_id = tokenizer.pad_token_id
        else:
            raise ValueError(f"NLU tasks currently only support qwen3 and llama, got {config.model_type}")
    else:
        # Load generation model for generation tasks
        if config.model_type == "llama" and args.adapter.lower() in [
            "mlora",
            "moelora",
            "samora",
            "homelora",
        ]:
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
                trust_remote_code=True,
            )
        elif config.model_type == "qwen3" and args.adapter.lower() in [
            "mlora",
            "moelora",
            "samora",
            "homelora",
        ]:
            model = Qwen3ForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
                trust_remote_code=True,
            ) 
        
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    if args.adapter.lower() == "mlora":
        mlora_config = {
            "type": "mlora",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lambda_num": args.lambda_num,
            "B_num": args.num_B,
            "B_scale": args.temperature,
            "diagonal_format": False,
        }
    elif args.adapter.lower() == "multilora":
        mlora_config = {
            "type": "multilora",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_num": args.lora_num,
        }
    elif args.adapter.lower() == "moelora":
        mlora_config = {
            "type": "moelora",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "expert_num": args.expert_num,
            "task_num": args.task_num,
            "task_embedding_dim": args.te_dim,
        }
    elif args.adapter.lower() == "dora":
        mlora_config = {
            "type": "dora",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "merge_weights": args.merge_weights,
            "Wdecompose": args.Wdecompose,
            "dora_simple": args.dora_simple,
        }
    elif args.adapter.lower() in ["samora", "homelora", "samora"]:
        mlora_config = {
            "type": "samora",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lambda_num": args.lambda_num,
            "B_num": args.num_B,
            "B_scale": args.temperature,
            "diagonal_format": args.diagonal_format,
            "tunable_scaler": args.tunable_scaler,
        }
    elif args.adapter.lower() == "lora":
        mlora_config = {
            "type": "lora",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "merge_weights": args.merge_weights,
        }
    elif args.adapter.lower() == "hydralora":
        mlora_config = {
            "type": "hydralora",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_num": args.lora_num,
            "B_scale": args.temperature if args.temperature is not None else 0.0,
        }

    model = wrap_model(model, args.lora_target_modules, mlora_config)
    
    state_dict = torch.load(lora_weights, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg.unexpected_keys)
    
    if is_nlu_task:
        heads_path = args.heads_path
        if heads_path is None:
            lora_dir = os.path.dirname(os.path.abspath(lora_weights))
            checkpoint_dir = os.path.join(lora_dir, "checkpoint")
            if os.path.exists(os.path.join(checkpoint_dir, "heads.pt")):
                heads_path = os.path.join(checkpoint_dir, "heads.pt")
            elif os.path.exists(os.path.join(lora_dir, "heads.pt")):
                heads_path = os.path.join(lora_dir, "heads.pt")
        
        if heads_path is not None and os.path.exists(heads_path):
            try:
                model_device = next(model.parameters()).device if list(model.parameters()) else torch.device(device)
                heads_state_dict = torch.load(heads_path, map_location=model_device)
                heads_state_dict = {k.replace("module.", ""): v for k, v in heads_state_dict.items()}
                
                head_msg = model.load_state_dict(heads_state_dict, strict=False)
                if len(head_msg.missing_keys) > 0:
                    print(f"Warning: Missing keys when loading heads: {head_msg.missing_keys}")
                if len(head_msg.unexpected_keys) > 0:
                    print(f"Warning: Unexpected keys when loading heads: {head_msg.unexpected_keys}")
                
                if hasattr(model, "score"):
                    model_device = next(model.model.parameters()).device if list(model.model.parameters()) else torch.device(device)
                    model.score = model.score.to(model_device)
                    print(f"Moved score layer to device: {model_device}")
                
                print(f"Successfully loaded classification heads from {heads_path}")
            except Exception as e:
                print(f"Warning: Failed to load classification heads from {heads_path}: {e}")
                import traceback
                traceback.print_exc()
        elif is_nlu_task:
            print(f"Warning: Classification heads not found. Expected at {heads_path or 'auto-detected path'}")
    
    print_parameter_count(state_dict, model, is_nlu_task=is_nlu_task)
    
    unexpected_keys = [k for k in msg.unexpected_keys if "heads" not in k.lower() and "score" not in k.lower()]
    if len(unexpected_keys) > 0:
        print(f"Warning: Unexpected keys (excluding heads/score): {unexpected_keys}")

    for name, param in model.named_parameters():
        param.requires_grad = False

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ""
    if not instruction:
        raise ValueError("instruct not initialized")
    return instruction


def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    if dataset == "boolq":
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"true|false", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == "piqa":
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"solution1|solution2", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ["social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa"]:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"answer1|answer2|answer3|answer4|answer5", sentence_)
        if not pred_answers:
            pred_answers = re.findall(r"1|2|3|4", sentence_)
            if not pred_answers:
                return ""
        return pred_answers[0]
    elif dataset == "hellaswag":
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"ending1|ending2|ending3|ending4", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == "winogrande":
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"option1|option2", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


if __name__ == "__main__":
    main()