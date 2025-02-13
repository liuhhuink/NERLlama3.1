#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NERLlama3.1
@IDE     ：PyCharm 
@Author  ：JEatC
@Date    ：2025
"""
from utils.tools import *


# ##check input and content equal
def check_input_equal_content(input, content):
    if content.replace('@@', '').replace('##', '') == input:
        return True
    else:
        return False


# ##format checking
def check_llm_result_format(json_file):
    # ##
    formatted = True
    count = 0
    # ##
    json_data = load_json_data_base(json_file)
    # ##
    for obj in json_data:
        input = obj['input']
        new_content = obj['new_content']
        if check_input_equal_content(input, new_content):
            pass
        else:
            formatted = False
            count += 1
            print('\033[93m### count {},{}, ner result format error {}.\033[0m'.format(count, get_current_time(),
                                                                                       new_content))
    if formatted:
        print("\033[92m***{},LLM NER formatted.\033[0m".format(get_current_time()))
    else:
        print("\033[92m***{},LLM NER format error, count {}.\033[0m".format(get_current_time(), count))
    return formatted


# ##convert ner result to BIO format
def convert_format_ner_result_to_BIO(src_file, save_file):
    # ##
    save_data = ''
    # ##
    if check_llm_result_format(src_file):
        # ##
        json_data = load_json_data_base(src_file)
        # ##
        for obj in json_data:
            input = obj['input']
            new_content = obj['new_content']
            flag = 0
            for i in range(len(input.split(' '))):
                if not flag:
                    if input.split(' ')[i] == new_content.split(' ')[i]:
                        save_data += (input.split(' ')[i] + '\tO\n')
                    elif new_content.split(' ')[i].startswith('@@') and new_content.split(' ')[i].endswith('##'):
                        save_data += (input.split(' ')[i] + '\tB\n')
                    elif new_content.split(' ')[i].startswith('@@'):
                        save_data += (input.split(' ')[i] + '\tB\n')
                        flag = True
                    elif new_content.split(' ')[i].endswith('##'):
                        save_data += (input.split(' ')[i] + '\tI\n')
                        flag = False
                    else:
                        print('Error1',input)
                        exit()
                else:
                    if input.split(' ')[i] == new_content.split(' ')[i]:
                        save_data += (input.split(' ')[i] + '\tI\n')
                    elif new_content.split(' ')[i].startswith('@@') and new_content.split(' ')[i].endswith('##'):
                        save_data += (input.split(' ')[i] + '\tB\n')
                    elif new_content.split(' ')[i].startswith('@@'):
                        save_data += (input.split(' ')[i] + '\tB\n')
                        flag = True
                    elif new_content.split(' ')[i].endswith('##'):
                        save_data += (input.split(' ')[i] + '\tI\n')
                        flag = False
                    else:
                        print('Error2')
                        exit()
            save_data += '\n'

        with open(save_file, 'w', encoding='utf-8') as save_f:
            save_f.write(save_data)
            save_f.close()
        print(
            '\033[91m***{}, NER Result format transformed and save to {}.\033[0m'.format(get_current_time(), save_file))


if __name__ == '__main__':
    # ##[dataset name]
    # dataset_name = 'NCBI-disease'
    dataset_name = 'BC5CDR-chem'
    # dataset_name = 'BC2GM-gene'
    # dataset_name = 'BC5CDR-disease'
    # dataset_name = 'NLM-gene'
    # dataset_name = 'NLM-chem'
    # dataset_name = 'BC4CHEMD-chem'
    # dataset_name = 'JNLPBA-gene'
    # dataset_name = 's800-species'
    # ##[data file path:ner result file, save bio format data file]
    src_file = fr'../data\ner_datasets\{dataset_name}\sft_test_ner_preprocessing_multi_task_final_manual.json'
    save_file = fr'../data\ner_datasets\{dataset_name}\sft_test_ner_bio_result_multi_task.txt'
    convert_format_ner_result_to_BIO(src_file, save_file)
