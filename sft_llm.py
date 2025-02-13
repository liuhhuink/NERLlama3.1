#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NERLlama3.1
@IDE     ：PyCharm 
@Author  ：JEatC
@Date    ：2025
"""
import torch
import random
from datasets import load_dataset, DatasetDict
from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments, AutoTokenizer
from transformers import BitsAndBytesConfig

# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# ##[definition of dataset path,base model path and fine-tuned model save path]
sft_model_path = r'local_models/meta-llama/Meta-Llama-3.1-8B-Instruct'
sft_model_save_path = r'checkpoints/Meta-Llama-3.1-8B-Instruct-sft-multitask'
sft_dataset_path = r'data/sft_datasets/multitask'

# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# ##[tokenizer initialization]
tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Set reasonable default for models without max length
if tokenizer.model_max_length > 100_000:
    tokenizer.model_max_length = 2048


# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# ##[dataset load and process]
def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example


# ##load
raw_datasets = load_dataset(sft_dataset_path)
dataset_dict = {"train": raw_datasets["train"]}

raw_datasets = DatasetDict(dataset_dict)
#
print('raw datasets', raw_datasets)
example = raw_datasets["train"][0]
messages = example["messages"]
for message in messages:
    role = message["role"]
    content = message["content"]
    print('{0:20}:  {1}'.format(role, content))
#
column_names = list(raw_datasets["train"].features)
raw_datasets = raw_datasets.map(apply_chat_template,
                                num_proc=1,
                                fn_kwargs={"tokenizer": tokenizer},
                                remove_columns=column_names,
                                desc="Applying chat template", )
train_test_split = raw_datasets['train'].train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']
for index in random.sample(range(len(raw_datasets["train"])), 3):
    print(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")
    print("#####################################")

# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# ##[llama3.1 supervised fine-tuning parameter setting]
# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

device_map = "auto"  # {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

# attn_implementation=False,#"flash_attention_2",
# set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
model_kwargs = dict(
    torch_dtype="auto",
    use_cache=False,  # set to False as we're going to use gradient checkpointing
    device_map=device_map,
    quantization_config=quantization_config,
)

# based on config
training_args = TrainingArguments(
    fp16=False,  # specify bf16=True instead when training on GPUs that support bf16 else fp16
    bf16=False,
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2.0e-05,
    log_level="info",
    logging_steps=5,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=5,
    output_dir=sft_model_save_path,
    overwrite_output_dir=True,
    per_device_eval_batch_size=1,  # originally set to 8
    per_device_train_batch_size=1,  # originally set to 8
    # push_to_hub=False,
    # hub_model_id=trained_model_id,
    # hub_strategy="every_save",
    # report_to="tensorboard",
    report_to="none",  # for skipping wandb logging
    save_strategy="no",
    save_total_limit=None,
    seed=42,
)

# based on config
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
# del trainer
torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=sft_model_path,
    model_init_kwargs=model_kwargs,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    packing=True,
    peft_config=peft_config,
    max_seq_length=tokenizer.model_max_length,
)

# To clear out cache for unsuccessful run
torch.cuda.empty_cache()

# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# ##[train and save model]
trainer.train()
trainer.save_model(sft_model_save_path)
