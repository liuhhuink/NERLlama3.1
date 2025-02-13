#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NERLlama3.1
@IDE     ：PyCharm 
@Author  ：JEatC
@Date    ：2025
"""


# ##get system message while generate llama3.1 supervised fine-tuning dataset
def get_system_message():
    sys_m = (r'Below is an instruction that describes a task, paired'
             r' with an input that provides further context. Write a'
             r' response that appropriately completes the request')
    return sys_m


# ##get user message head while generate llama3.1 supervised fine-tuning dataset
def get_user_instruction_head(ent_type):
    user_instruction_head = (r'You are an high intelligent biomedical named entity recognition tool, '
                             r'extract \"{}\" entities from the given text, only mark the entity on '
                             r'the original input by highlighting them with @@ and ##, do not make '
                             r'any other changes. if there is no entity recognized, output the same sentence.\n'
                             r'Below are some examples:\n')
    return user_instruction_head.format(ent_type)


# ##get input cautious message while generate llama3.1 supervised fine-tuning dataset
def get_user_input_cautious():
    input_cautious = 'Be cautious here is the Input:\n'
    return input_cautious
