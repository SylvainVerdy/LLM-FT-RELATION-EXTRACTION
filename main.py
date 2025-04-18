# -*- coding: utf-8 -*-

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig,
    get_cosine_schedule_with_warmup,
    set_seed,

)
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
import time
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, Trainer, GenerationConfig
from accelerate import Accelerator
from torch.utils.data.distributed import DistributedSampler

import os
# disable Weights and Biases
os.environ['WANDB_DISABLED']="true"

from transformers import AutoTokenizer
import json
from datasets import load_dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

access_token = "HF_KEY"

def tokens_to_text(sentences):
    def join_tokens(tokens):
        text = ""
        for i, token in enumerate(tokens):
            if i > 0:
                # Check if the current token is a punctuation mark
                if token in [',', '.', '!', '?', ':', ';', ')', ']', '}']:
                    text += token
                # Check if the current or previous token is a dash
                elif token == '-' or tokens[i-1] == '-':
                    text += token
                # Check if the previous token ended with an opening parenthesis or bracket
                elif tokens[i-1] in ['(', '[', '{']:
                    text += token
                # Check for apostrophes in contractions
                elif token.startswith("'"):
                    text += token
                else:
                    text += ' ' + token
            else:
                text += token
        return text

    # Join tokens in each sentence
    joined_sentences = [join_tokens(sentence) for sentence in sentences]
    # Join sentences with a space
    text = ' '.join(joined_sentences)
    return text


def prompt_extraction_relations(text, predefined_relations_set, number_of_entity_pairs, entity_pairs):
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a
                response that appropriately completes the request.
                ### Instruction:
    Your task is to determine whether there are relations between the entity pairs based on the information in
    the text. If there exists relations, select relations for the entity pairs from the relation set; if there is no
    relation, return None.

    The format of the input entity pair is ‘(head entity| -| tail entity)’.
    Your output format is ‘(head entity| relation/None| tail entity)’.

    ### Relation set:
    {predefined_relations_set}

    ### Text:
    {text}
    ### {number_of_entity_pairs} Entity pairs:
    {entity_pairs}
    """

    return prompt


# Prepare the dataset
def prepare_dataset(examples):
    inputs = []
    labels = []
    rels = json.load(open("/content/rel_info.json", "r"))
    for i in range(len(examples["sents"])):
        text = tokens_to_text(examples["sents"][i])
        predefined_relations_set = ", ".join(rels)

        # for key, value in rels.items():
        #     predefined_relations_set.append((f"{key}:{value}"))
        entity_pairs, labels_list = [], []

        for o in range(len(examples["labels"][i]["head"])):
            head = examples["vertexSet"][i][examples["labels"][i]["head"][o]][0]["name"]
            tail = examples["vertexSet"][i][examples["labels"][i]["tail"][o]][0]["name"]
            entity_pairs.append(f"({head}|-|{tail})")
            labels_list.append(f"({head}|{examples['labels'][i]['relation_id'][o]}|{tail})")
        number_of_entity_pairs = len(entity_pairs)
        entity_pairs_str = "\n".join(entity_pairs)

        labels_str = "\n".join(labels_list)

        # Generate prompts
        prompt = prompt_extraction_relations(text, predefined_relations_set, number_of_entity_pairs, entity_pairs_str)
        prompt += "\n### Response:" + labels_str

        # Append prompts to the list
        inputs.append(prompt)

        # Generate labels (you may need to adjust this based on your specific requirements)

        labels.append(labels_str)

    return {"input": inputs, "output": labels}

def read_file(path):
    return json.load(open(path, "r"))

def preprocess_data(tokenizer): #config_yaml_dataset, tokenizer):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # train = read_file(config_yaml_dataset['train'])
    # dev = read_file(config_yaml_dataset['dev'])
    dataset = load_dataset("docred")

    def tokenize_function(examples):
        model_inputs = tokenizer(examples["input"], return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
        labels = tokenizer(examples["output"], return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    dataset = load_dataset("docred")
    # First mapping
    tokenized_datasets = dataset.map(prepare_dataset, batched=True) # remove_columns=["validation", "test", "train_annotated", "train_distant"])
    # tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True, remove_columns=["title", "sents", "vertexSet"]) #"input", "output"])

    return tokenized_datasets


model_name='microsoft/Phi-3-mini-128k-instruct'

tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,padding_side="left",add_eos_token=True,add_bos_token=True,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

tokenized_datasets = preprocess_data(tokenizer)

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

accelerator = Accelerator()

device_map = {"": 0}
original_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map=device_map,
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=True, token=access_token)


# 'target_modules' is a list of the modules that should be targeted by LoRA.
target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]

# 'set_seed(1234)' sets the random seed for reproducibility.
set_seed(1234)

config = LoraConfig(
    r=32, #Rank
    lora_alpha=32,
    target_modules=target_modules,
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

# 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
original_model.gradient_checkpointing_enable()
original_model.enable_input_require_grads()

# 2 - Prepare model for training
peft_model = get_peft_model(original_model, config)

print(tokenized_datasets)

output_dir = f'./peft-docred-re-training-{str(int(time.time()))}'
import transformers

peft_training_args = TrainingArguments(
    output_dir = output_dir,
    warmup_ratio=0.3,
    max_grad_norm=0.3,
    gradient_accumulation_steps=4,
    per_device_train_batch_size=1,
    num_train_epochs=5,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    logging_steps=25,
    logging_dir="./logs",
    # save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=25,
    do_eval=True,
    gradient_checkpointing=True,
    report_to="none",
    overwrite_output_dir = 'True',
    group_by_length=True,
    local_rank=accelerator.local_process_index,
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
)

peft_model.config.use_cache = False

response_template = "\n### Response:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


peft_trainer = SFTTrainer(
    model=peft_model,
    train_dataset=tokenized_datasets["train_annotated"],
    eval_dataset=tokenized_datasets["validation"],
    peft_config=config,
    dataset_text_field="input",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=peft_training_args,
    data_collator=collator,
    # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)


peft_trainer, training_dataloader, eval_dataloader = accelerator.prepare(
    peft_trainer, peft_trainer.get_train_dataloader(), peft_trainer.get_eval_dataloader()
)


import torch
torch.cuda.empty_cache()
# peft_trainer.train()
# peft_trainer.model.save_pretrained("/content/llm-re")
# Replace peft_trainer.train() with:
with accelerator.main_process_first():
    peft_trainer.train()

# Ensure we're on the main process for saving
if accelerator.is_main_process:
    peft_trainer.model.save_pretrained("/content/llm-re")

