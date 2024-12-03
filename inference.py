import json
import sys
import os
import fire
import torch
import time
import transformers
import numpy as np
from typing import List
from peft.peft_model import set_peft_model_state_dict
from loraprune.peft_model import get_peft_model
from loraprune.utils import freeze, prune_from_checkpoint
from loraprune.lora import LoraConfig
from datasets import load_dataset
from dataset_types import MedicalReport

from transformers import AutoModelForCausalLM, AutoTokenizer
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from transformers import pipeline
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    base_model: str = "",
    dataset: str = "",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.,
    lora_target_modules: List[str] = [
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj"
        ],
    lora_weights: str = "tloen/alpaca-lora-7b",
    cutoff_len: int = 128
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    assert (
        dataset
    ), "Please specify a --dataset, e.g. --dataset='wikitext'"

    tokenizer = AutoTokenizer.from_pretrained(base_model, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    hf_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer, device_map='auto')

    model = get_peft_model(model, config)
    if lora_weights:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            lora_weights, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                lora_weights, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            for name, param in adapters_weights.items():
                if 'lora_mask' in name:
                    adapters_weights[name] = param.reshape(-1)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model = model.to(device)

    freeze(model)
    prune_from_checkpoint(model)

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.half()  # seems to fix bugs for some users.


    model.eval()
    
    #########################################################
    # Run on dataset
    #########################################################

    data = load_dataset("json", data_files=dataset)["train"]

    hf_pipeline.model = model

    results = []
    # Process data in batches
    for i, sample in tqdm(enumerate(data), desc="Processing..."):
        # Create a character level parser and build a transformers prefix function
        parser = JsonSchemaParser(MedicalReport.schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(hf_pipeline.tokenizer, parser)

        # Process batch
        outputs = hf_pipeline(sample["prompt"], prefix_allowed_tokens_fn=prefix_function, max_length=4096)

        # Extract results
        result = outputs[0]['generated_text'][len(sample["prompt"]):]
        medical_report = MedicalReport.model_validate_json(result)
        if not medical_report:
            print(f"Invalid medical report: {medical_report}")
        results.append({str(i): medical_report.model_dump_json()})

    with open('medical_report_output.json', 'w') as f:
        json.dump(results, f)

    return


if __name__ == "__main__":
    fire.Fire(main)