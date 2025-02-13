#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NERLlama3.1
@IDE     ：PyCharm 
@Author  ：JEatC
@Date    ：2025
"""
import seqeval.scheme
from seqeval.metrics import f1_score, precision_score, recall_score


# ##load bio format data
def load_ner_data(src_file, tag='disease'):
    """
    :param src_file:
    :param tag: disease/chemical/gene
    :return:
     [
        ['O', 'O', 'B-PER', 'I-PER', 'O'],
        ['O', 'B-LOC', 'I-LOC', 'B-LOC', 'I-LOC', 'O'],
        ...
    ]
    """
    #
    ents = []
    src_f = open(src_file, 'r', encoding='utf-8')
    lines = src_f.readlines()
    tmp = []
    for line in lines:
        if line == '\n':
            ents.append(tmp)
            tmp = []
        else:
            label = line.strip('\n').split('\t')[1]
            if label == 'O':
                tmp.append(label)
            else:
                tmp.append(label + '-' + tag)
                # tmp.append(label)
    ents.append(tmp)
    return ents


if __name__ == '__main__':
    # ##
    # dataset_name = 'NCBI-disease'
    dataset_name = 'BC5CDR-chem'
    # dataset_name = 'BC2GM-gene'
    # dataset_name = 'BC5CDR-disease'
    # dataset_name = 'NLM-gene'
    # dataset_name = 'NLM-chem'
    # ##
    true_file = fr'../data/ner_datasets/{dataset_name}/test.tsv'
    pred_file = fr'../data/ner_datasets/{dataset_name}/sft_test_ner_bio_result_multi_task.txt'
    # pred_file = fr'../data/ner_datasets/{dataset_name}/test_bern2_ner.txt'
    # ##
    true_ents = load_ner_data(true_file)
    pred_ents = load_ner_data(pred_file)

    #
    precision = precision_score(true_ents, pred_ents, average='micro')
    recall = recall_score(true_ents, pred_ents, average='micro')
    f1 = f1_score(true_ents, pred_ents, average='micro')
    # ##
    print("\033[91mPrecision: {}\033[0m".format(precision))
    print("\033[92mRecall: {}\033[0m".format(recall))
    print("\033[94mF1_Score: {}\033[0m".format(f1))
