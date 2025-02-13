#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：BioNER-LLaMA3
@IDE     ：PyCharm 
@Author  ：JEatC
@Date    ：2025
"""
from tools import *


# ## dataset statistics
def ner_dataset_instance_mention_statistic(file):
    # ##
    instance = 0
    mentions = 0
    # ##
    json_data = load_json_data_base(file)
    # ##
    instance = len(json_data)
    # ##
    for obj in json_data:
        output = obj['output']
        count = output.count('@@')
        mentions += count
    print('\033[91m###{}, Instance {}, Mentions {}, Dataset Name {}.\033[0m'.format(get_current_time(), instance,
                                                                                    mentions, file))


if __name__ == '__main__':
    # ##
    # dataset_name = 'NCBI-disease'
    # dataset_name = 'BC5CDR-chem'
    # dataset_name = 'BC2GM-gene'
    # dataset_name = 'BC5CDR-disease'
    # dataset_name = 'NLM-chem'
    dataset_name = 'NLM-gene'
    # ##
    # file_name = 'test.json'
    # file_name = 'devel.json'
    file_name = 'train_dev.json'
    # ##
    src_file = fr'../data/ner_datasets/{dataset_name}/{file_name}'
    # ##
    ner_dataset_instance_mention_statistic(file=src_file)
