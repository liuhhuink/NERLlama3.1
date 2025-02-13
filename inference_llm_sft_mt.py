#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：BioNER-LLaMA3
@IDE     ：PyCharm 
@Author  ：JEatC
@Date    ：2025
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
from transformers import BitsAndBytesConfig
from utils.tools import *
import random
from config.config import *

# torch.manual_seed(None)
# **************************************************************************
# **************************************************************************
# **************************************************************************
# ##load model
base_model_id = "local_models/meta-llama/Meta-Llama-3.1-8B-Instruct"
new_model_id = "checkpoints/Meta-Llama-3.1-8B-Instruct-sft-multitask-v2"
device_map = "auto"

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id,
                                                  quantization_config=quantization_config,
                                                  device_map=device_map)
model = PeftModel.from_pretrained(base_model, new_model_id)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


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
                             eos_token_id=terminators, do_sample=True, temperature=1, top_p=0.95, top_k=30)
                             # eos_token_id=terminators, do_sample=True, temperature=1, top_p=0.95, top_k=30)
                             # eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
    # ##
    response = outputs[0][input_ids.shape[-1]:]

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


user_instruction_head = (r'You are an high intelligent biomedical named entity recognition tool, '
                         r'extract "disease","chemical" and "gene" entities from the given text, '
                         r'only mark the entity on the original input by highlighting them with '
                         r'@disease@ and #disease#,@chem@ and #chem#, @gene@ and #gene#, '
                         r'do not make any other changes. if there is no entity recognized, '
                         r'output the same sentence.\n'
                         r'Below are some examples:\n')


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
    res_message = user_instruction_head
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
    count = 0
    # ##
    ref_obj_data = load_json_data_base(ref_file)
    # ##
    json_data = load_json_data_base(src_file)
    ent_type = src_file.split('/')[-2].split('-')[-1]
    for obj in json_data:
        no = obj['no']
        input = obj['input']
        output = obj['output']
        # ner_result_old = obj['ner_result_old']
        # ##
        request_message = request_message_generation(input=input, select_samples=random.sample(ref_obj_data, 3),
                                                     ent_type=ent_type)
        # ##
        ner_result = local_llama3_model_request(request_message)
        count += 1
        # ##
        # ##
        print('\033[92m###{}, count: {}. extract result: {}.\033[0m'.format(get_current_time(), count, ner_result))
        # ##
        tmp = {}
        tmp['no'] = no
        tmp['input'] = input
        tmp['output'] = output
        # tmp['ner_result_old'] = ner_result_old
        tmp['ner_result'] = ner_result
        save_data.append(tmp)
    # ##
    with open(save_file, 'w', encoding='utf-8') as save_f:
        json.dump(save_data, save_f, ensure_ascii=False, indent=4)
        save_f.close()


if __name__ == '__main__':
    # ##[dataset name]
    # dataset_name = 'BC5CDR-disease'
    dataset_name = 'BC2GM-gene'

    # ##[data file path:test set file, train and dev file, save ner result file]
    src_file = f'../data/ner_datasets/{dataset_name}/test.json'
    ref_file = f'../data/ner_datasets/Multi-Task/train_dev_multi_task.json'
    save_file = f'../data/ner_datasets/{dataset_name}/sft_test.json'
    # ##[extract named entity]
    local_test_dataset_and_do_ner(src_file=src_file, ref_file=ref_file, save_file=save_file)
