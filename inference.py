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
from peft import LoraConfig
from datasets import load_from_disk
from dataset_types import MedicalReport

from transformers import LlamaForCausalLM, LlamaTokenizer
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from transformers import pipeline

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

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
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
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    from torch.utils.data.dataset import Dataset
    times = []
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)

    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    def PPLMetric(model, loader, device="cuda"):
        ppl = llama_eval(model, loader, device)
        print(ppl)
        return ppl

    @torch.no_grad()
    def llama_eval(model, loader, device):
        model.eval()
        nlls = []
        n_samples = 0
        for batch in loader:
            batch = batch.to(device)
            with torch.cuda.amp.autocast():
                t1 = time.time()
                output = model(batch)
                times.append(time.time() - t1)
            lm_logits = output.logits

            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss)
        # print(torch.cat(nlls, dim=-1).mean())
        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        return ppl.item()
    
    #########################################################
    # Run on dataset
    #########################################################

    data = load_from_disk(dataset)

    hf_pipeline.model = model

    results = []
    for i, sample in enumerate(data):
        prompt = """
Extract the medical report information into the following model:
{schema}
If something is not clear, or incomplete, leave it blank.

### INPUT:
{instruction}

### RESPONSE:
""".format(
            schema=MedicalReport.schema(),
            instruction=sample["instruction"]
        )

        # Create a character level parser and build a transformers prefix function from it
        parser = JsonSchemaParser(MedicalReport.schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(hf_pipeline.tokenizer, parser)

        # Call the pipeline with the prefix function
        output_dict = hf_pipeline(prompt, prefix_allowed_tokens_fn=prefix_function, max_length=4096)

        # Extract the results
        result = output_dict[0]['generated_text'][len(prompt):]
        # Save result to file instead of printing
        results.append({str(i): result})
    
    with open('medical_report_output.json', 'w') as f:
        json.dump(results, f)

    return


if __name__ == "__main__":
    fire.Fire(main)