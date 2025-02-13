#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NERLlama3.1
@IDE     ：PyCharm 
@Author  ：JEatC
@Date    ：2025
"""
import os
import time
import json
import pandas as pd


#
def get_current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


# 创建文件夹
def create_new_dir(save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    else:
        print('### {}  {} .'.format(get_current_time(), save_dir))


# ##读取json文件
def load_json_data_base(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.read()
        json_data = json.loads(data)
        f.close()
        return json_data


# ##dict保存为parquet
def dict_save_to_parquet(data_dict, save_file):
    df = pd.DataFrame(data_dict)
    df.to_parquet(save_file, engine='pyarrow')
    print('###{},dict data save to {}.'.format(get_current_time(), save_file))
