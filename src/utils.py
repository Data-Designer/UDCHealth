# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : utils.py
# Time       ：3/7/2024 11:29 am
# Author     ：XXXXXX
# version    ：python 
# Description：一些工具函数； https://pythonhosted.org/PyMedTermino/tuto_en.html
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch
import pickle
import gzip

from sklearn.manifold import TSNE
from pyhealth.medcode.codes.atc import ATC
from pyhealth.medcode import InnerMap
from pyhealth.datasets import SampleBaseDataset
from pyhealth.tokenizer import Tokenizer
from itertools import chain
from typing import Optional, Tuple, Union, List
import itertools




def set_random_seed(seed):
    """ 设置随机种子以确保代码的可重复性 """
    random.seed(seed)       # Python 内置的随机库
    np.random.seed(seed)    # NumPy 库
    torch.manual_seed(seed) # PyTorch 库

    # 如果您使用 CUDA，则还需要添加以下两行代码
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU


def cal_num(all_samples):
    total_label = []
    for sample in all_samples:
        total_label.append(sample['label'])
    # print(len(tota_drug)) # 有部分药物从未出现过；
    return len(total_label), sum(total_label)



def convert_to_multihot(batch_data, num_classes):
    """
    将 B*M 的用户历史交互数据转换为 B*K 的 multi-hot 编码。

    参数:
    - batch_data: 形状为 (B, M) 的张量，记录用户的历史交互。
    - num_classes: 可能的交互项类别总数 K。

    返回:
    - 形状为 (B, K) 的 multi-hot 编码张量。
    """
    B, M = batch_data.shape

    # 初始化一个全零的 multi-hot 编码矩阵，形状为 (B, K)
    multi_hot = torch.zeros(B, num_classes, device=batch_data.device)

    # 使用 scatter_ 函数将对应位置的值设置为 1
    multi_hot.scatter_(1, batch_data, 1)

    return multi_hot


def get_last_visit_sample(samples):
    """提取sample中的最后一次就诊记录"""
    last_visits = {}
    for record in samples:
        patient_id = record['patient_id']
        visit_id = int(record['visit_id'])  # 将visit_id转换为整数
        if patient_id not in last_visits or visit_id > int(last_visits[patient_id]['visit_id']):
            last_visits[patient_id] = record
    print("Patient Number: ", len(last_visits))
    return last_visits


def ana_rare_disease(samples, rare_disease):
    """添加标签"""
    rare_disease = set(dict(rare_disease).keys())
    for record in samples:
        # rare为1
        for i in range(len(record['conditions_wc'])): # [[2,3,4],[1,2,3]]
            record['conditions_wc'][i] = [str(float(item in rare_disease)) for item in record['conditions_wc'][i]]
    return samples

def generate_rare_disease(samples, threshold, path, task='DIAG', mode='code'):
    """生成稀有疾病, train中少于threshold的疾病; 这里rare disease的定义是出现code的次数"""
    # 先找到last visit
    last_visits = get_last_visit_sample(samples).values()

    disease_num = {} # 记录数据中每个疾病的数量, 本来想加label的
    disease_num_copy = {} # 这里仅记录train中见过的疾病，即评测和替换的时候都要放上
    if mode == 'code': # 会遇到困扰，因为很多病人持续的获得一些病证
        if task == 'DIAG':
            for record in last_visits:
                for disease_lis in record['conditions']:#  + [record['labels']]: # nest list
                    for disease in disease_lis:
                        if disease not in disease_num:
                            disease_num[disease] = 1
                        else:
                            disease_num[disease] += 1
            for record in last_visits:
                for disease_lis in record['conditions']  + [record['labels']]: # nest list
                    for disease in disease_lis:
                        if disease not in disease_num_copy:
                            disease_num_copy[disease] = 1
                        else:
                            disease_num_copy[disease] += 1

        elif task == 'REC':
            for record in last_visits:
                for disease_lis in record['conditions']:
                    for disease in disease_lis:
                        if disease not in disease_num:
                            disease_num[disease] = 1
                        else:
                            disease_num[disease] += 1
    # elif mode == 'patient':
    #     if task == "DIAG":
    #         for record in last_visits:
    #             conditions = record['conditions']# + [record['labels']]
    #             conditions = list(set(itertools.chain.from_iterable(conditions)))
    #             for disease in conditions:
    #                 if disease not in disease_num:
    #                     disease_num[disease] = 1
    #                 else:
    #                     disease_num[disease] += 1
    #     elif task == "REC":
    #         for record in last_visits:
    #             conditions = list(set(itertools.chain.from_iterable(record['conditions']))) # 同一个病人只统计一次。
    #             for disease in conditions:
    #                 if disease not in disease_num:
    #                     disease_num[disease] = 1
    #                 else:
    #                     disease_num[disease] += 1

    # 按数量对字典进行排序
    sorted_items = sorted(disease_num.items(), key=lambda x: x[1], reverse=False) # {'disease':16,'dsa':18 升序}
    # 计算需要选取的item数量
    num_items_to_select = int(np.ceil(threshold * len(sorted_items)))
    num_train_drl = int(np.ceil((1-threshold) * len(sorted_items)))

    # 选取前30%的item及其数量
    tail_percent_items = sorted_items[:num_items_to_select] # Figure 1 num_train_drl, 这里是前80%的数据

    rarest_percent_items = sorted_items[:num_train_drl] # 用于对齐训练，只用头部去对齐, 有点少了啊， 这里是排除了前20%的数据 (而且是训练而非全部)
    top_percent_items = sorted_items[num_items_to_select:] # 最common的数据

    file = {
        'most_common_disease': top_percent_items,
        'all_disease': sorted_items,
        'filter_drl_items': rarest_percent_items, # 过滤名单rarest
        'rare_disease': tail_percent_items # 用于替换rare
    }
    cal_top = sum([i[1] for i in top_percent_items])
    cal_tail = sum([i[1] for i in tail_percent_items])
    print("ALLDiag {} ALL interactions {}".format(len(sorted_items),sum([i[1] for i in sorted_items]))) # # rec: 2089503, top. 3887,1984264, tail:3888,3888
    print('cal top train_drl tail num  {}, top tail interaction num: {}'.format((len(top_percent_items), len(tail_percent_items)), (cal_top, cal_tail)))


    # 生成不同的group
    if task=='DIAG':
        sorted_items_copy = sorted(disease_num_copy.items(), key=lambda x: x[1], reverse=False) # {'disease':16,'dsa':18 升序}
        num_items_to_select_copy = int(np.ceil(threshold * len(sorted_items_copy)))
        num_train_drl_copy = int(np.ceil((1 - threshold) * len(sorted_items_copy)))
        tail_percent_items_copy = sorted_items_copy[:num_items_to_select_copy] # Figure 1 num_train_drl, 这里是前80%的数据
        rarest_percent_items_copy = sorted_items_copy[:num_train_drl_copy] # 用于对齐训练，只用头部去对齐, 有点少了啊， 这里是排除了前20%的数据 (而且是训练而非全部)
        top_percent_items_copy = sorted_items_copy[num_items_to_select_copy:] # 最common的数据
        sorted_items = sorted_items_copy # 很多只在label, 所以可以这么做


    group_file = {}
    percentile_ranges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for start, end in zip(percentile_ranges[:-1], percentile_ranges[1:]):
        start_index = int(start * len(sorted_items)) # 这里放copy，不然真的难搞。
        end_index = int(end * len(sorted_items))
        group_file[f"{start * 100:.0f}%-{end * 100:.0f}%"] = [k for k, _ in sorted_items[start_index:end_index]]
    file['group_disease'] = group_file

    save_pickle(file, path + 'rare.pkl')
    print("rare disease generate done!")

    return file


def generate_rare_patient(samples, disease_group, path):
    """for rec"""
    last_visits = get_last_visit_sample(samples).values()
    # print(disease_group)
    group_patient = {}
    for record in last_visits:
        patient_id = record['patient_id']
        conditions = set(itertools.chain.from_iterable(record['conditions']))
        for key, disease_set in disease_group.items(): # 从最稀少开始
            if len(conditions & set(disease_set)) > 0:
                group_patient[patient_id] = key
                break
    # new_group_patient = {}
    # for key, patient_ls in group_patient.items():
    #     for patient in patient_ls:
    #         if patient not in new_group_patient:
    #             new_group_patient[patient] = [key]
    #         else:
    #             new_group_patient[patient].append(key)
    # print("AAAA", group_patient)
    save_pickle(group_patient, path + 'rare_patient.pkl')
    print("rare patient id generate done!")
    return




def pad_nested_list(tokens, pad_value=0):
    """[[[1,0,3],[3,0]],[[1,0,3],[3,0],[4,0,0]]]"""
    # 获取最内层和外层的最大长度
    max_len_inner = max(len(inner) for outer in tokens for inner in outer)
    max_len_outer = max(len(outer) for outer in tokens)

    # 初始化填充值矩阵
    padded_tokens = np.full((len(tokens), max_len_outer, max_len_inner), pad_value, dtype=object)

    # 填充矩阵
    for i, outer in enumerate(tokens):
        for j, inner in enumerate(outer):
            padded_tokens[i, j, :len(inner)] = inner

    return padded_tokens.tolist()

def get_tokenizers(dataset, special_tokens=False):
    if not special_tokens:
        special_tokens = ["<unk>", "<pad>"] # 把pad取消
    feature_keys = ["conditions", "procedures", "drugs"]
    feature_tokenizers = {}
    for feature_key in feature_keys:
        feature_tokenizers[feature_key] = Tokenizer(
            tokens=dataset.get_all_tokens(key=feature_key),
            special_tokens=special_tokens,
        )
        print(feature_key, feature_tokenizers[feature_key].get_vocabulary_size())
    return feature_tokenizers


import torch

def get_last_visit(hidden_states, mask):
    """Gets the last visit from the sequence model.

    Args:
        hidden_states: [batch size, seq len, hidden_size]
        mask: [batch size, seq len]

    Returns:
        last_visit: [batch size, hidden_size]
    """
    if mask is None:
        return hidden_states[:, -1, :]
    else:
        mask = mask.long()

        # 检查每一行是否全为False，如果是，则将其处理为只考虑最后一个元素
        all_false_rows = torch.all(mask == 0, dim=1)
        if torch.any(all_false_rows):
            mask[all_false_rows, -1] = 1

        last_visit = torch.sum(mask, 1) - 1
        last_visit = last_visit.unsqueeze(-1)
        last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
        last_visit = torch.reshape(last_visit, hidden_states.shape)
        last_hidden_states = torch.gather(hidden_states, 1, last_visit)
        last_hidden_state = last_hidden_states[:, 0, :]
        return last_hidden_state


# def get_last_visit(hidden_states, mask):
#     """Gets the last visit from the sequence model.
#
#     Args:
#         hidden_states: [batch size, seq len, hidden_size]
#         mask: [batch size, seq len]
#
#     Returns:
#         last_visit: [batch size, hidden_size]
#     """
#     if mask is None:
#         return hidden_states[:, -1, :]
#     else:
#         mask = mask.long()
#         last_visit = torch.sum(mask, 1) - 1
#         last_visit = last_visit.unsqueeze(-1)
#         last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
#         last_visit = torch.reshape(last_visit, hidden_states.shape)
#         last_hidden_states = torch.gather(hidden_states, 1, last_visit)
#         last_hidden_state = last_hidden_states[:, 0, :]
#         return last_hidden_state


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'wb') as f:
        data = pickle.dump(data, f)
    print("File has beeen saved to {}.".format(file_path))
    return

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()







def get_atc_name(level):
    level = level + 1
    code_sys = ATC(refresh_cache=True)  # 第一次需要
    name_map = {}
    for index in code_sys.graph.nodes:
        if len(index) == level:
            name = code_sys.graph.nodes[index]['name']
            name_map[index] = name
    return name_map



def get_aux_icd(feature_key):
    """有些old icd找不到"""
    if feature_key == 'conditions':
        colspecs = [(0, 5), (6, 14), (15, 16), (17, 77), (78, None)]
        # Read the data into a DataFrame
        df = pd.read_fwf('/home/xxxx/xxxx/data/icd10cm_order_2016.txt', colspecs=colspecs, header=None)
        # Assign column names
        df.columns = ['ID', 'Code', 'Flag', 'Description', 'Add']
        df['Description'] = df['Description'].apply(lambda x: x.split(',')[0])

        df_trimmed = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        dic = df_trimmed.set_index('Code')['Description'].to_dict()
    else:
        colspecs = [(0, 8), (8, None)]
        # Read the data into a DataFrame
        df = pd.read_fwf('/home/xxxx/xxxx/data/icd10pcs_codes_2016.txt', colspecs=colspecs, header=None)
        df2 = pd.read_fwf('/home/xxxx/xxxx/data/icd10pcs_codes_2017.txt', colspecs=colspecs, header=None)
        df3 = pd.read_fwf('/home/xxxx/xxxx/data/icd10pcs_codes_2021.txt', colspecs=colspecs, header=None)
        # Assign column names
        df.columns = ['Code', 'Description']
        df2.columns = ['Code', 'Description']
        df3.columns = ['Code', 'Description']
        df['Description'] = df['Description'].apply(lambda x: x.split(',')[0])
        df2['Description'] = df2['Description'].apply(lambda x: x.split(',')[0])
        df3['Description'] = df3['Description'].apply(lambda x: x.split(',')[0])

        df_trimmed = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df2_trimmed = df2.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df3_trimmed = df3.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        # print(df_trimmed.head())
        dic = df_trimmed.set_index('Code')['Description'].to_dict()
        dic2 = df2_trimmed.set_index('Code')['Description'].to_dict()
        dic3 = df3_trimmed.set_index('Code')['Description'].to_dict()
        dic.update(dic2)
        dic.update(dic3)
    return dic


def get_stand_system(dataset):
    """返回三个编码系统，不然太慢了"""
    if dataset=='MIMIC-III':
        diag_sys = InnerMap.load("ICD9CM")
        proc_sys = InnerMap.load("ICD9PROC")
        med_sys = ATC(refresh_cache=False)
    else:
        diag_sys = InnerMap.load("ICD10CM")
        proc_sys = InnerMap.load("ICD10PROC")
        med_sys = ATC(refresh_cache=False)
    return diag_sys, proc_sys, med_sys


def get_node_name(code_type, reverse_stand=True):
    """for ICD9CM-DIAG, for ICD9PROC, 这样可以搞定standard"""
    code_sys = InnerMap.load(code_type)
    name_map = {}
    for index in code_sys.graph.nodes:
        name = code_sys.graph.nodes[index]['name']
        name_map[index] = name
    if reverse_stand:
        name_map = {key.replace('.', ''): value for key, value in name_map.items()}  # [{ATC, name}]
    return name_map


def get_name_map(config):
    diag_id2_name, proc_id2_name, drug_id2_name = {}, {}, {}
    if config['DATASET'] == 'MIII':
        name_map_diag = get_node_name('ICD9CM')
        name_map_proc = get_node_name('ICD9PROC')
        name_map_med = get_atc_name(config['ATCLEVEL'])
        diag_id2_name.update(name_map_diag)
        proc_id2_name.update(name_map_proc)
        drug_id2_name.update(name_map_med)
    elif config['DATASET'] == 'MIV':
        name_map_diag = get_node_name('ICD9CM')  # 这里需要同时替换所有的ID, 为ICD9和10分别处理一份吧，别慌。 居然有些ID没有
        name_map_proc = get_node_name('ICD9PROC')
        name_map_med = get_atc_name(config['ATCLEVEL'])
        name_map_diag2 = get_node_name('ICD10CM')  # 只有600多重叠，很少，即使有不对的也可以当噪声
        name_map_proc2 = get_node_name('ICD10PROC')
        name_map_med2 = get_atc_name(config['ATCLEVEL'])
        diag_id2_name.update(name_map_diag)
        diag_id2_name.update(name_map_diag2)
        proc_id2_name.update(name_map_proc)
        proc_id2_name.update(name_map_proc2)
        drug_id2_name.update(name_map_med)
        drug_id2_name.update(name_map_med2)

        intersection = set(name_map_diag.keys()) & set(name_map_diag2.keys())

        print("partial overlap {} between ICD9 and ICD10, but it's ok.".format(len(intersection)))
    elif config['DATASET'] == 'eICU':
        # name_map_diag = get_node_name('ICD9CM')  # 这里需要同时替换所有的ID, 为ICD9和10分别处理一份吧，别慌。 居然有些ID没有
        # name_map_diag2 = get_node_name('ICD10CM')  # 只有600多重叠，很少，即使有不对的也可以当噪声
        # diag_id2_name.update(name_map_diag)
        # diag_id2_name.update(name_map_diag2)
        # name_map_diag = get_node_name('ICD9CM')  # 无med code
        # name_map_proc = get_node_name('ICD9PROC')
        data = pd.read_csv('/home/xxxxxxx/HyperHealth/data/physionet.org/files/eicu-crd/2.0/diagnosis.csv',
                           header=0,
                           index_col=None,
                           dtype={'icd9code': str, 'diagnosisstring': str},
                           )
        data = data.dropna(subset=["icd9code", "diagnosisstring"])
        # data['icd9code'] = data['icd9code'].apply(lambda x: x.split(',')[0].strip()) # 分割取第一个
        name_map_diag ={}
        for index, row in data.iterrows():
            icd9codes = row['icd9code'].split(',')
            for icd9code in icd9codes:
                name_map_diag[icd9code.strip()] = row['diagnosisstring']
        diag_id2_name.update(name_map_diag)
        print("EICU处理成功")
        # proc_id2_name.update(name_map_proc)
   

    return diag_id2_name, proc_id2_name, drug_id2_name


def split_by_patient(
        dataset: SampleBaseDataset,
        ratios: Union[Tuple[float, float, float], List[float]],
        train_ratio=1.0,
        seed: Optional[int] = None,
        warm_cold: bool = False,
):
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    patient_indx = list(dataset.patient_to_index.keys()) # 存储数据 {patientID: [index]}
    num_patients = len(patient_indx)
    np.random.shuffle(patient_indx)
    train_patient_indx = patient_indx[: int(num_patients * ratios[0])]
    np.random.seed(seed)
    np.random.shuffle(train_patient_indx)
    train_patient_indx = train_patient_indx[: int(len(train_patient_indx) * train_ratio)]
    val_patient_indx = patient_indx[
                       int(num_patients * ratios[0]): int(
                           num_patients * (ratios[0] + ratios[1]))
                       ]
    test_patient_indx = patient_indx[int(num_patients * (ratios[0] + ratios[1])):]
    train_index = list(
        chain(*[dataset.patient_to_index[i] for i in train_patient_indx])
    )
    val_index = list(chain(*[dataset.patient_to_index[i] for i in val_patient_indx]))
    test_index = list(chain(*[dataset.patient_to_index[i] for i in test_patient_indx]))

    min_length = min(len(lst) for lst in dataset.patient_to_index.values())
    print("最短列表的长度为:", min_length)

    if warm_cold:
        warm_patient_index = []
        cold_patient_index = []
        # 这里放一些东西
        for i in test_patient_indx:
            patient_index = dataset.patient_to_index[i] # lis
            if len(patient_index) > 1: # 最少是1数据来着
                warm_patient_index.extend(patient_index)
            else:
                cold_patient_index.extend(patient_index)
        if warm_cold == 'warm':
            test_dataset = torch.utils.data.Subset(dataset, warm_patient_index)
        elif warm_cold == 'cold':
            test_dataset = torch.utils.data.Subset(dataset, cold_patient_index)
    else:
        test_dataset = torch.utils.data.Subset(dataset, test_index)

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    # test_dataset = torch.utils.data.Subset(dataset, test_index)
    return train_dataset, val_dataset, test_dataset


def split_by_patient_one(
        dataset: SampleBaseDataset,
        ratios: Union[Tuple[float, float, float], List[float]],
        train_ratio=1.0,
        seed: Optional[int] = None,
        warm_cold: bool = False,
):
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    patient_indx = list(dataset.patient_to_index.keys()) # 存储数据 {patientID: [index]}
    num_patients = len(patient_indx)
    np.random.shuffle(patient_indx)
    train_patient_indx = patient_indx[: int(num_patients * ratios[0])]
    np.random.seed(seed)
    np.random.shuffle(train_patient_indx)
    train_patient_indx = train_patient_indx[: int(len(train_patient_indx) * train_ratio)]
    val_patient_indx = patient_indx[
                       int(num_patients * ratios[0]): int(
                           num_patients * (ratios[0] + ratios[1]))
                       ]
    test_patient_indx = patient_indx[int(num_patients * (ratios[0] + ratios[1])):]
    train_index = list(
        chain(*[dataset.patient_to_index[i] for i in train_patient_indx])
    )
    val_index = list(chain(*[dataset.patient_to_index[i] for i in val_patient_indx]))
    test_index = list(chain(*[dataset.patient_to_index[i] for i in test_patient_indx]))

    min_length = min(len(lst) for lst in dataset.patient_to_index.values())
    # print("最短列表的长度为:", min_length)

    if warm_cold:
        warm_patient_index = []
        cold_patient_index = []
        # 这里放一些东西
        for i in test_patient_indx:
            patient_index = dataset.patient_to_index[i] # lis
            if len(patient_index) > 1: # 最少是1数据来着
                warm_patient_index.extend(patient_index)
            else:
                cold_patient_index.extend(patient_index)
        test_dataset_warm = torch.utils.data.Subset(dataset, warm_patient_index)
        test_dataset_cold = torch.utils.data.Subset(dataset, cold_patient_index)
    else:
        test_dataset_warm = torch.utils.data.Subset(dataset, test_index)
        test_dataset_cold = None

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    # test_dataset = torch.utils.data.Subset(dataset, test_index)
    return train_dataset, val_dataset, test_dataset_warm, test_dataset_cold

def achieve_samples(dataset):
    """subset没有办法获取samples,或者直接重构subset方法
    https://www.cnblogs.com/orion-orion/p/15906086.html"""
    samples = []
    for i in range(len(dataset)):
        samples.append(dataset[i])
    return samples




def plot_tsne(embedding_layer, title='t-SNE Visualization'):
    """
    Plot t-SNE visualization of node embeddings for a PyTorch embedding layer.

    Parameters:
        embedding_layer (torch.nn.Embedding): PyTorch embedding layer.
        title (str): Title of the plot (default is 't-SNE Visualization').

    Returns:
        None
    """
    # 获取嵌入层的权重（嵌入矩阵）
    embedding_matrix = embedding_layer.data.cpu().numpy()

    # 使用 sklearn 的 t-SNE 进行降维
    tsne = TSNE(n_components=2)
    embedding_2d = tsne.fit_transform(embedding_matrix)

    # 绘制 t-SNE 图像
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], marker='.')
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    # plt.show()
    plt.savefig('/home/xxxxx/HyperHealth/draw/img/'+ title + '.pdf', format='pdf', dpi=300)


def decompress_gz_files(directory):
    # 获取目录下所有文件
    files = os.listdir(directory)

    # 遍历每个文件
    for file in files:
        # 检查文件是否为.gz文件
        if file.endswith('.gz'):
            file_path = os.path.join(directory, file)
            output_path = os.path.splitext(file_path)[0]  # 去除.gz后缀

            # 打开.gz文件并解压到新文件
            with gzip.open(file_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())

            print(f"解压文件 '{file_path}' 到 '{output_path}'")


def f1_cal(pred, ground):
    # 转换为集合用于计算
    pred_set = set(pred)
    ground_set = set(ground)
    # 真正例 (TP): 预测和实际都是正的
    tp = len(pred_set.intersection(ground_set))
    print("TP", pred_set.intersection(ground_set))
    # 假正例 (FP): 预测为正但实际为负
    fp = len(pred_set - ground_set)
    print("FP", pred_set - ground_set)
    # 假负例 (FN): 预测为负但实际为正
    fn = len(ground_set - pred_set)
    print("FN", ground_set - pred_set)
    # 计算精确度和召回率
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    # 计算 F1 分数
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1_score, fp, fn

def jaccard(pred, ground):
    pred_set = set(pred)
    ground_set = set(ground)
    # 计算 Jaccard 相似度
    jaccard = len(pred_set.intersection(ground_set)) / len(pred_set.union(ground_set)) if len(pred_set.union(ground_set)) > 0 else 0
    return jaccard



"""adjust_learning_rate"""
def lr_poly(base_lr, iter, max_iter, power):
    if iter > max_iter:
        iter = iter % max_iter
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, config, max_iter):
    lr = lr_poly(config['LR'], i_iter, max_iter, 0.9) # power=0.9
    optimizer.param_groups[0]['lr'] = np.around(lr,5)
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def merge_dicts(dict1, dict2, prefix1='pcf_', prefix2='plm_'):
    merged_dict = {}

    # 添加前缀/后缀以防止 key 冲突
    for key, value in dict1.items():
        new_key = prefix1 + str(key)
        merged_dict[new_key] = value

    for key, value in dict2.items():
        new_key = prefix2 + str(key)
        merged_dict[new_key] = value

    return merged_dict



def pad_batch(embs, lengths):
    """reshape a list into a batch"""
    lengths = np.asarray(lengths)
    cum_lengths = np.cumsum(lengths)
    batch_lis = [embs[i - l:i] for i, l in zip(cum_lengths, lengths)]

    batch_lis = torch.nn.utils.rnn.pad_sequence(batch_lis, batch_first=True, padding_value=0) # [torch.randn(4,8), torch.randn(3,8)]->[torch.randn(2,4,8)]
    return batch_lis

def shift_padding(mask):
    shifted_mask = torch.zeros_like(mask)  # 创建一个与原始 mask 矩阵相同形状的零矩阵
    shifted_mask[:, :-1] = mask[:, 1:]  # 将原始 mask 矩阵向左移动一位，右边用零填充
    return shifted_mask.to(mask.device)



def get_indices(adjacency_matrix):
    """邻接矩阵的非零"""
    nonzero_column_indices = torch.nonzero(adjacency_matrix, as_tuple=False)[:, 1]
    lens = list(torch.sum(adjacency_matrix, dim=1).long().cpu().numpy())
    return nonzero_column_indices, lens

def get_nonzero_values(matrix):
    """其他非0 indice"""
    nonzero_values = torch.masked_select(matrix, matrix != 0)
    return nonzero_values

def create_interaction_matrix(interactions, total_drugs):
    """
    创建用户与药物的交互矩阵。

    参数:
    interactions : torch.Tensor
        用户与药物的交互历史索引矩阵，形状为 [B, D]，其中 -1 表示无交互。
    total_drugs : int
        药物的总数。

    返回:
    torch.Tensor
        布尔矩阵，形状为 [B, M]，标记用户与药物的交互。
    """
    B, D = interactions.shape  # 用户数和每个用户的最大交互数
    # 初始化用户*药物矩阵，初始值为False
    user_drug_matrix = torch.zeros((B, total_drugs), dtype=torch.bool).to(interactions.device)

    # 替换-1为有效索引，因为-1不是有效的索引，我们暂时转换为0
    valid_interactions = interactions.clone()
    valid_interactions[interactions == -1] = 0

    # 使用scatter_更新matrix，将对应的位置设为True
    user_drug_matrix.scatter_(1, valid_interactions, 1)

    # 将无效的0索引（原来的-1）重新标记为False，仅当原始数据中该位置为-1时
    if (interactions == -1).any():
        user_drug_matrix[:, 0] = user_drug_matrix[:, 0] & (interactions != -1)

    return user_drug_matrix


def build_map(b_map, max=None, config=None):
    """根据b-map向b-map_进行插值"""
    batch_size, b_len = b_map.size()
    if max is None:
        max = b_map.max() + 1
    if config['USE_CUDA']:
        b_map_ = torch.cuda.FloatTensor(batch_size, b_len, max).fill_(0).to(b_map.device)
    else:
        b_map_ = torch.zeros(batch_size, b_len, max)
    b_map_.scatter_(2, b_map.unsqueeze(2), 1.)
    b_map_.requires_grad=False
    return b_map_

if __name__ == '__main__':
    pass
    # decompress_gz_files('/home/xxxxx/HyperHealth/data/physionet.org/files/picdb/1.1.0')
