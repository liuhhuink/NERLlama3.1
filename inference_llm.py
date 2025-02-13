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
from utils.tools import *
from config.config import *
from transformers import AutoTokenizer, AutoModelForCausalLM

# **************************************************************************
# **************************************************************************
# **************************************************************************
# ##load model
model_id = "local_models/meta-llama/Meta-Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
# **************************************************************************
# **************************************************************************
# **************************************************************************
# ##tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Set reasonable default for models without max length
if tokenizer.model_max_length > 100_000:
    tokenizer.model_max_length = 2048

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


# **************************************************************************
# **************************************************************************
# **************************************************************************
# ##local Meta-Llama3.1-8B-Instruct model request
def local_llama3_model_request(messages):
    # ##
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
        model.device)
    # ##
    outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_new_tokens=128,
                             eos_token_id=terminators, do_sample=True, temperature=0.6,
                             top_p=0.9, )
    # ##
    response = outputs[0][input_ids.shape[-1]:]
    # ##
    # print('\033[92m###{}, extract result: {}.\033[0m'.format(get_current_time(),
    #                                                          tokenizer.decode(response, skip_special_tokens=True)))
    return tokenizer.decode(response, skip_special_tokens=True)


# ##few shot examples generation
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


# ##request message generation
def request_message_generation(input, select_samples, ent_type):
    """
    :param input:
    :param select_samples:
    :param ent_type:
    :return:
    [   {"role":"system", "content":""}
        {"role":"user", "content":""},
    ]
    """
    # ##
    res_message = get_user_instruction_head(ent_type)
    # ##
    few_shot_examples = few_shot_examples_generation(select_samples=select_samples)
    # ##
    res_message += few_shot_examples
    # ##
    res_message += get_user_input_cautious() + input
    return [{"role": "system", "content": get_system_message()},
            {"role": "user", "content": res_message}]


# ##load ner test dataset and extract entity
def local_test_dataset_and_do_ner(src_file, ref_file, save_file):
    # ##
    save_data = []
    # ##
    ref_obj_data = load_json_data_base(ref_file)
    # ##
    json_data = load_json_data_base(src_file)
    # ##
    ent_type = src_file.split('/')[-2].split('-')[-1]
    count = 0
    for obj in json_data:
        no = obj['no']
        input = obj['input']
        output = obj['output']
        # ##
        request_message = request_message_generation(input=input, select_samples=random.sample(ref_obj_data, 3),
                                                     ent_type=ent_type)
        # ##
        ner_result = local_llama3_model_request(request_message)
        # ##
        count += 1
        print('\033[92m###{}, count: {}. extract result: {}.\033[0m'.format(get_current_time(), count, ner_result))
        # ##
        tmp = {}
        tmp['no'] = no
        tmp['input'] = input
        tmp['output'] = output
        tmp['ner_result'] = ner_result
        save_data.append(tmp)
    # ##
    with open(save_file, 'w', encoding='utf-8') as save_f:
        json.dump(save_data, save_f, ensure_ascii=False, indent=4)
        save_f.close()


if __name__ == '__main__':
    dataset_name = 'ncbi-disease'

    # ##[data file path:test set file, train and dev file, save ner result file]
    src_file = f'../data/ner_datasets/{dataset_name}/test.json'
    ref_file = f'../data/ner_datasets/{dataset_name}/train_dev.json'
    save_file = f'../data/ner_datasets/{dataset_name}/test_origin.json'
    # ##[extract named entity]
    local_test_dataset_and_do_ner(src_file=src_file, ref_file=ref_file, save_file=save_file)
