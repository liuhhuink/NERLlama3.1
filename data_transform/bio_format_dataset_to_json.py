#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NERLlama3.1
@IDE     ：PyCharm 
@Author  ：JEatC
@Date    ：2025
"""
from utils.tools import *


# ##
def check_input_equal_content(input, content):
    if content.replace('@@', '').replace('##', '') == input:
        return True
    else:
        return False


# ##
def check_llm_result_format(json_file):
    # ##
    formatted = True
    count = 0
    # ##
    json_data = load_json_data_base(json_file)
    # ##
    for obj in json_data:
        input = obj['input']
        new_content = obj['output']
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


# ##
def word_label_list_to_sentence(word_list, label_list):
    sentence = ' '.join(word_list)
    sentence_labeled = ''
    # ##
    for i in range(len(label_list)):
        if label_list[i] == 'O':
            sentence_labeled += (word_list[i] + ' ')
        else:
            if label_list[i] == 'B' and i < len(label_list) - 1:
                if label_list[i + 1] == 'I':
                    sentence_labeled += ('@@' + word_list[i] + ' ')
                else:
                    sentence_labeled += ('@@' + word_list[i] + '## ')
            elif label_list[i] == 'B' and i == len(label_list) - 1:
                sentence_labeled += ('@@' + word_list[i] + ' ')
            elif label_list[i] == 'I' and i < len(label_list) - 1:
                if label_list[i + 1] != 'I':
                    sentence_labeled += (word_list[i] + '## ')
                else:
                    sentence_labeled += (word_list[i] + ' ')
            elif label_list[i] == 'I' and i == len(label_list) - 1:
                sentence_labeled += (word_list[i] + '## ')
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
    # ##format check
    check_llm_result_format(save_file)


if __name__ == '__main__':
    # ##[dataset name]
    # dataset_name = 'NCBI-disease'
    # dataset_name = 'BC5CDR-chem'
    # dataset_name = 'BC2GM-gene'
    # dataset_name = 'BC5CDR-disease'
    # dataset_name = 'NLM-chem'
    dataset_name = 'NLM-gene'

    # dataset_name = 's800-species'
    # dataset_name = 'linnaeus-species'

    # ##[file name]
    # file_name = 'train'
    # file_name = 'test'
    file_name = 'devel'
    # file_name = 'train_dev'

    # ##[bio format data file path]
    src_file = fr'../data/ner_datasets/{dataset_name}/{file_name}.tsv'

    # ##[save json data file path]
    save_file = src_file[:-4] + '.json'

    # ##[bio format data transform to json data]
    bio_format_ner_dataset_to_json(src_file, save_file)
