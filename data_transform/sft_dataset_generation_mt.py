#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：BioNER-LLaMA3
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
from config.config import get_system_message, get_user_input_cautious


# <editor-folder desc='data transform'>

def multi_task_dataset_generation_from_bio_format(chem_file, disease_file, save_file):
    # ##
    word_list = []
    chem_label_list = []
    disease_label_list = []
    # ##
    with open(chem_file, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line == '\n':
                # ##
                chem_label_list.append('\n')
                word_list.append('\n')
            else:
                word_list.append(line.strip().split('\t')[0])
                chem_label_list.append(line.strip().split('\t')[1])
        f.close()
    # ##
    with open(disease_file, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line == '\n':
                # ##
                disease_label_list.append('\n')
            else:
                disease_label_list.append(line.strip().split('\t')[1])
        f.close()
    assert len(word_list) == len(chem_label_list) == len(disease_label_list)
    save_data = ''
    for i in range(len(word_list)):
        # ##
        if word_list[i] == '\n':
            save_data += ('\n')
        else:
            if chem_label_list[i] != 'O':
                save_data += (word_list[i] + '\t' + chem_label_list[i] + '-chem\n')
            elif disease_label_list[i] != 'O':
                save_data += (word_list[i] + '\t' + disease_label_list[i] + '-disease\n')
            else:
                save_data += (word_list[i] + '\tO\n')
    save_f = open(save_file, 'w', encoding='utf-8')
    save_f.write(save_data)


# ##
def bio_format_label_transform(src_file, save_file):
    # ##
    word_list = []
    label_list = []
    save_data = ''
    # ##
    with open(src_file, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line == '\n':
                # ##
                label_list.append('\n')
                word_list.append('\n')
            else:
                word_list.append(line.strip().split('\t')[0])
                label_list.append(line.strip().split('\t')[1] + '-gene')
        f.close()
    for i in range(len(word_list)):
        # ##
        if word_list[i] == '\n':
            save_data += ('\n')
        else:
            if label_list[i] != 'O-gene':
                save_data += (word_list[i] + '\t' + label_list[i] + '\n')
            else:
                save_data += (word_list[i] + '\tO\n')
    save_f = open(save_file, 'w', encoding='utf-8')
    save_f.write(save_data)


# ##
def word_label_list_to_sentence(word_list, label_list):
    sentence = ' '.join(word_list)
    sentence_labeled = ''
    # ##
    for i in range(len(label_list)):
        if label_list[i] == 'O':
            sentence_labeled += (word_list[i] + ' ')
        else:
            if label_list[i].startswith('B') and i < len(label_list) - 1:
                if label_list[i + 1].startswith('I'):
                    sentence_labeled += ('@' + label_list[i].split('-')[1] + '@' + word_list[i] + ' ')
                else:
                    sentence_labeled += ('@' + label_list[i].split('-')[1] + '@' + word_list[i] + '#' +
                                         label_list[i].split('-')[1] + '# ')
            elif label_list[i].startswith('B') and i == len(label_list) - 1:
                # print(label_list[i].split('-'))
                sentence_labeled += ('@' + label_list[i].split('-')[1] + '@' + word_list[i] + ' ')
            elif label_list[i].startswith('I') and i < len(label_list) - 1:
                if not label_list[i + 1].startswith('I'):
                    sentence_labeled += (word_list[i] + '#' + label_list[i].split('-')[1] + '# ')
                else:
                    sentence_labeled += (word_list[i] + ' ')
            elif label_list[i].startswith('I') and i == len(label_list) - 1:
                sentence_labeled += (word_list[i] + '#' + label_list[i].split('-')[1] + '# ')
            else:
                print(word_list, label_list)
                print('error word label list to sentence')
                exit()
    # print(sentence)
    # print(sentence_labeled[:-1])
    return sentence, sentence_labeled[:-1]


# ##
def bio_format_ner_dataset_to_json(src_file, save_file):
    # ##
    save_data = []
    no = 0
    # ##
    with open(src_file, 'r', encoding='utf-8') as f:
        word_list = []
        label_list = []
        while True:
            line = f.readline()
            if not line:
                break
            if line == '\n':
                # ##
                sentence, sentence_labeled = word_label_list_to_sentence(word_list, label_list)
                no += 1
                tmp = {}
                tmp['no'] = no
                tmp['input'] = sentence
                tmp['output'] = sentence_labeled
                save_data.append(tmp)
                word_list = []
                label_list = []
            else:
                word_list.append(line.strip().split('\t')[0])
                label_list.append(line.strip().split('\t')[1])
        f.close()
    # ##
    with open(save_file, 'w', encoding='utf-8') as save_f:
        json.dump(save_data, save_f, ensure_ascii=False, indent=4)
        save_f.close()
    print(
        '###{},dataset ***{}** format transform and save to ***{}***.'.format(get_current_time(), src_file, save_file))


# </editor-folder>

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


def message_generation(user_message, assistant_message):
    sys_m = {'content': get_system_message(), 'role': 'system'}
    user_m = {'content': user_message, 'role': 'user'}
    assistant_m = {'content': assistant_message, 'role': 'assistant'}
    return [sys_m, user_m, assistant_m]


user_instruction_head = (r'You are an high intelligent biomedical named entity recognition tool, '
                         r'extract "disease","chemical" and "gene" entities from the given text, '
                         r'only mark the entity on the original input by highlighting them with '
                         r'@disease@ and #disease#,@chem@ and #chem#, @gene@ and #gene#, '
                         r'do not make any other changes. if there is no entity recognized, '
                         r'output the same sentence.\n'
                         r'Below are some examples:\n')


# ##[convert json file ner data to llama3 sft format]
def ner_sft_format_data_generation(json_data):
    # ##target format data
    save_data = []
    # ##load json file
    # json_data = load_json_data_base(json_file)

    # ##
    for obj in json_data:
        input = obj['input']
        output = obj['output']
        prompt = ''
        # prompt_head + few_shot_examples_generation(random.sample(json_data, 5)) + prompt_input_caution)
        prompt_id = prompt_id_generation(prompt)
        few_shot_examples = few_shot_examples_generation(random.sample(json_data, 5))
        messages = message_generation(user_instruction_head + few_shot_examples + get_user_input_cautious() + input,
                                      output)
        tmp = {}
        tmp['prompt'] = prompt
        tmp['messages'] = messages
        tmp['prompt_id'] = prompt_id
        save_data.append(tmp)
    return save_data


def multi_task_sft_dataset_generation(dataset_paths, save_file_path):
    # ##save file generation
    create_new_dir(save_file_path)
    create_new_dir(os.path.join(save_file_path, 'data'))
    save_file = os.path.join(os.path.join(save_file_path, 'data'), 'train.parquet')

    # ##[save data]
    # ##load data
    json_data = []
    for path in dataset_paths:
        json_data += (load_json_data_base(path))
    random.shuffle(json_data)
    sft_data = ner_sft_format_data_generation(json_data)
    # ##save data
    dict_save_to_parquet(sft_data, save_file)
    print('###{},llama3.1 supervised fine-tuning data generation and save to {}.'.format(get_current_time(), save_file))


if __name__ == '__main__':
    # ##
    file_name = 'test_chem_disease.tsv'
    # ##
    # chem_dataset_file = fr'../data/ner_datasets/BC5CDR-chem/{file_name}'
    # disease_dataset_file = fr'../data/ner_datasets/BC5CDR-disease/{file_name}'
    # save_file = disease_dataset_file[:-5] + '_chem_disease.tsv'
    # # ##
    # multi_task_dataset_generation_from_bio_format(chem_dataset_file, disease_dataset_file, save_file=save_file)
    # src_file = fr'../data/ner_datasets/BC2GM-gene/{file_name}'
    # save_file = src_file[:-4] + '_gene.tsv'
    # # ##
    # bio_format_label_transform(src_file, save_file)

    # ##
    # src_file = fr'../data/ner_datasets/Multi-Task/{file_name}'
    # save_file = src_file[:-4] + '.json'
    # bio_format_ner_dataset_to_json(src_file, save_file)

    # ##[dataset file path list]
    dataset_paths = [r'../data/ner_datasets/Multi-Task/train_dev_chem_disease.json',
                     '../data/ner_datasets/Multi-Task/train_dev_gene.json']
    # ##[save dataset file path]
    save_file_path = r'../data/sft_datasets/multitask'
    multi_task_sft_dataset_generation(dataset_paths, save_file_path)
