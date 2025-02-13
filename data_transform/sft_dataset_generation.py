#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NERLlama3.1
@IDE     ：PyCharm 
@Author  ：JEatC
@Date    ：2025
Meta-Llama3.1-8B-Instruct supervised fine-tuning data format
[{
    "prompt":"",
    "message":[{"content":"","role":"system"},{"content":"","role":"user"},{"content":"","role":"assistant"}
    ],
    "prompt_id":""
},
...
]
"""
import random
import hashlib
from utils.tools import *
from config.config import *


# ##[generate 64-bit prompt hash id]
def prompt_id_generation(prompt):
    # ##
    hash_object = hashlib.sha256(prompt.encode())
    prompt_id = hash_object.hexdigest()
    return prompt_id


# ##[generate few-shot examples]
def few_shot_examples_generation(select_samples):
    # ##
    few_shot_samples = ''
    # ##
    for i in range(len(select_samples)):
        input = select_samples[i]['input']
        output = select_samples[i]['output']
        few_shot_samples += ('Input:\n' + input + '\n')
        few_shot_samples += ('Output:\n' + output + '\n')
    return few_shot_samples


# ##[message generation]
def message_generation(user_message, assistant_message):
    sys_m = {'content': get_system_message(), 'role': 'system'}
    user_m = {'content': user_message, 'role': 'user'}
    assistant_m = {'content': assistant_message, 'role': 'assistant'}
    return [sys_m, user_m, assistant_m]


# ##[convert json file ner data to llama3 sft format]
def ner_sft_format_data_generation(json_file):
    # ##entity type
    ent_type = json_file.split('/')[-2].split('-')[-1]
    # ##target format data
    save_data = []
    # ##load json file
    json_data = load_json_data_base(json_file)

    # ##
    for obj in json_data:
        input = obj['input']
        output = obj['output']
        prompt = ''
        # prompt_head + few_shot_examples_generation(random.sample(json_data, 5)) + prompt_input_caution)
        prompt_id = prompt_id_generation(prompt)
        few_shot_examples = few_shot_examples_generation(random.sample(json_data, 5))
        messages = message_generation(
            get_user_instruction_head(ent_type) + few_shot_examples + get_user_input_cautious() + input, output)
        tmp = {}
        tmp['prompt'] = prompt
        tmp['messages'] = messages
        tmp['prompt_id'] = prompt_id
        save_data.append(tmp)
    return save_data


# ##convert ner data to llama3 supervised fine-tuning dataset
def llama3_sft_dataset_generation(dataset_paths, save_file_path):
    # ##save file generation
    create_new_dir(save_file_path)
    create_new_dir(os.path.join(save_file_path, 'data'))
    save_file = os.path.join(os.path.join(save_file_path, 'data'), 'train.parquet')

    # ##[save data]
    sft_data_all = []
    # ##load data
    for path in dataset_paths:
        sft_data = ner_sft_format_data_generation(path)
        sft_data_all += sft_data
    # ##save data
    dict_save_to_parquet(sft_data_all, save_file)
    print('###{},llama3.1 supervised fine-tuning data generation and save to {}.'.format(get_current_time(), save_file))


if __name__ == '__main__':
    # ##[dataset file path list]
    dataset_paths = [r'../data/ner_datasets/NCBI-disease/train_dev.json',
                     '../data/ner_datasets/BC5CDR-chem/train_dev.json',
                     '../data/ner_datasets/BC2GM-gene/train_dev.json']
    # ##[save dataset file path]
    save_file_path = r'../data/sft_datasets/hybrid'

    # ##[convert ner data to llama3 supervised fine-tuning dataset]
    llama3_sft_dataset_generation(dataset_paths, save_file_path)
