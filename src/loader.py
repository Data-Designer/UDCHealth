# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : loader.py
# Time       ：3/7/2024 11:29 am
# Author     ：XXXXX
# version    ：python 
# Description： 这个loader需要对DRL进行处理 pyhealth version is important
"""
import numpy as np
import torch

from utils import get_tokenizers, get_name_map, get_last_visit_sample
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import random
import itertools


def collate_fn_dict(batch):
    return {key: [d[key] for d in batch] for key in batch[0]} # conditions: [B,V,M]

# def masked_code():
#     pass

def random_mask_word(seq, mask_prob=0.15):
    mask_idx = '<unk>' # 为1
    for i, _ in enumerate(seq):
        prob = random.random()
        # mask token with 15% probability
        if prob < mask_prob: # 这个比例或许可以改一下
            prob /= mask_prob
            # 80% randomly change token to mask token
            if prob < 0.8:
                seq[i] = mask_idx
            # # 10% randomly change token to random token
            # elif prob < 0.9:
            #     seq[i] = random.choice()[1] # 可以替换为同batch
            else:
                pass
        else:
            pass
    return seq

def raremed_mask_nsp_collate_fn(batch):
    """这里启发了一些KG-based的算法可以在这里查询对应的triple set"""
    for index, d in enumerate(batch):
        batch[index]['conditions'] = list(batch[index]['conditions'][-1])# 这里没毛病烙铁，因为我们之前data构建的时候经过遍历, 性能据查
        batch[index]['procedures'] = list(batch[index]['procedures'][-1])  # [[]]->[]
        batch[index]['drugs_hist'] = list(batch[index]['drugs_hist'][-1])  # [[]]->[]
        # batch[index]['conditions'] = list(itertools.chain(*batch[index]['conditions']))
        # batch[index]['procedures'] = list(itertools.chain(*batch[index]['procedures'])) # [[]]->[]
        # batch[index]['drugs_hist'] = list(itertools.chain(*batch[index]['drugs_hist'])) # [[]]->[]
    data = {key: [d[key] for d in batch] for key in batch[0]}
    # data['m_labels'] = [d['conditions'] + d['procedures'] for d in batch] # [[cond, proc]],可以到时候组
    data['m_conditions'] = [random_mask_word(d['conditions']) for d in batch]
    data['m_procedures'] = [random_mask_word(d['procedures']) for d in batch]
    # data['pairs'] = [(d['conditions'], d['procedures']) for d in batch] # 也可以到时候组
    # data['n_procedures'] = [random.sample(data['procedures'][:i] + data['procedures'][i + 1:], 1)[0] for i in range(len(data['conditions']))]
    return data








def get_dataloader(dataset, batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_dict):
    """for first, third stage"""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last
    )

    return dataloader


class DRLDataset(Dataset):
    """generative pairs for DRL training"""
    def __init__(self, disease, sample_dataset, train_dataset, config):
        cf_data, co_data, ref_matrix, cond_proc_matrix, cond_drug_matrix, diag_text, proc_text, drug_text = self.pre_dataset(disease, sample_dataset, train_dataset, config)
        self.diag_text = diag_text
        self.cond_proc_text = proc_text
        self.cond_drug_text = drug_text
        self.cf_data = cf_data # []
        self.co_data = co_data # []
        self.ref_matrix = ref_matrix # 对应的交互数据,遍历得到的， 按顺序排好
        self.cond_proc_matrix = cond_proc_matrix
        self.cond_drug_matrix = cond_drug_matrix
        self.ref_k = config['REFK'] # ground truth: negative
        self.cond_k = config['CONDK'] # condition-k

    def pre_dataset(self, disease, sample_dataset, train_dataset, config):
        """generate pairs for DRL traininig; eICU估计得单独处理"""
        # get tokenizers
        # rare_disease = dict(disease['rare_disease']).keys()
        rare_disease = dict(disease['filter_drl_items']).keys() # 0.8训练
        tokenizer = get_tokenizers(sample_dataset, special_tokens=['<unk>', '<pad>']) # 保持相同的tokenizer
        diag_voc_all = tokenizer['conditions'].vocabulary.token2idx # {'480.1': 2}; 不对啊，他不能使用diseaseDE ,因为label里的他看不见
        proc_voc = tokenizer['procedures'].vocabulary.token2idx # {'480.1': 2}
        drug_voc = tokenizer['drugs'].vocabulary.token2idx # {'480.1': 2}

        # get name map, 这里获取name还是要慎重些，可能会有问题
        diag_id2_name, proc_id2_name, drug_id2_name = get_name_map(config) # {'408.1':'content'}
        # print(diag_voc_all)
        if config['DATASET']=='eICU' or 'OMOP' or 'PIC' or 'OMIX': # drug不是标准的编码，
            drug_id2_name = tokenizer['drugs'].vocabulary.idx2token
        if config['DATASET'] == 'OMOP': #  OMOP感觉是concept,使用text话omop不能用感觉,omop全是非标准
            diag_id2_name = tokenizer['conditions'].vocabulary.idx2token # omop不大对
        if config['DATASET'] == 'eICU':
            proc_id2_name = tokenizer['procedures'].vocabulary.idx2token # 他的procedure也很烦人


        # get diag_text & cond_text, code得去掉., 这里得仔细检查下

        all_diag_name = [diag_id2_name.get(code, 'blank') for code in diag_voc_all.keys()]
        all_proc_name = [proc_id2_name.get(code, 'blank') for code in proc_voc.keys()]
        all_drug_name = [drug_id2_name.get(code, 'blank') for code in drug_voc.keys()]


        # get ref_matrix & cond_matrix
        diag2diag, diag2proc, diag2drug, diag2nexdrug = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int)) ,defaultdict(lambda: defaultdict(int)) # 给定数量，sampling {x: (code, num)}
        last_visits = get_last_visit_sample(train_dataset).values() # {'patient_id': {'conditions': [], 'procedures': [], 'drugs_hist': []}
        for patient in last_visits: # 后面还得做一个sample的处理（dataset for inference ）
            for visit in range(len(patient['conditions'])):
                diags, procs, drugs = patient['conditions'][visit], patient['procedures'][visit],patient['drugs_hist'][visit]
                if len(diags) * len(procs) * len(drugs)==0:
                    print("AAAAAAA")
                if config['TASK'] == "DIAG":
                    # next_diags = patient['labels']
                    # next_diags = patient['conditions'][visit + 1] if visit + 1 < len(patient['conditions']) else []
                    if visit + 1 < len(patient['conditions']):
                        next_diags = patient['conditions'][visit + 1]
                    elif visit + 1 == len(patient['conditions']):
                        next_diags = [i for i in patient['labels'] if i in set(diag_voc_all.keys())] # 增加样本量

                    for diag in diags:
                        for dia in next_diags: # next;
                            diag2diag[diag][dia] += 1
                        for proc in procs:
                            diag2proc[diag][proc] += 1 # 不能用zip，不等长
                        for drug in drugs:
                            diag2drug[diag][drug] += 1
                    ref_matrix = np.zeros((len(diag_voc_all), len(diag_voc_all)))

                elif config['TASK'] == "REC":
                    # next_drugs = patient['drugs_hist'][visit + 1] if visit + 1 < len(patient['drugs_hist']) else []
                    # next_drugs = patient['labels']
                    if visit + 1 < len(patient['drugs_hist']):
                        next_drugs = patient['drugs_hist'][visit + 1]
                    elif visit + 1 == len(patient['drugs_hist']):
                        next_drugs = [i for i in patient['labels'] if i in set(drug_voc.keys())] # 增加样本量

                    for diag in diags:
                        for dia in diags: # next
                            diag2diag[diag][dia] += 1
                        for proc in procs:
                            diag2proc[diag][proc] += 1 # 不能用zip，不等长
                        for drug in drugs:
                            diag2drug[diag][drug] += 1
                        for drug in next_drugs:
                            diag2nexdrug[diag][drug] += 1

                    ref_matrix = np.zeros((len(diag_voc_all), len(drug_voc)))

        cond_proc_matrix = np.zeros((len(diag_voc_all), len(proc_voc)))
        cond_drug_matrix = np.zeros((len(diag_voc_all), len(drug_voc)))

        for diag in diag2diag.keys():
            if config['TASK'] == 'DIAG':
                for dia in diag2diag[diag]:
                    ref_matrix[diag_voc_all[diag], diag_voc_all[dia]] = diag2diag[diag][dia] # 这里可以直接放数字
            elif config['TASK'] == 'REC':
                for drug in diag2nexdrug[diag]:
                    ref_matrix[diag_voc_all[diag], drug_voc[drug]] = diag2nexdrug[diag][drug]
            for proc in diag2proc[diag]:
                cond_proc_matrix[diag_voc_all[diag], proc_voc[proc]] = diag2proc[diag][proc]
            for drug in diag2drug[diag]:
                cond_drug_matrix[diag_voc_all[diag], drug_voc[drug]] = diag2drug[diag][drug]

        # for diag in diag2diag.keys():
        #     if cond_drug_matrix[diag_voc_all[diag]].sum()==0:
        #         print("BBBBBB", diag)

        ref_matrix[:2, :] = 0 # <unk>, <pad> = 0
        ref_matrix[:, :2] = 0 # # <unk>, <pad> = 0
        np.fill_diagonal(ref_matrix, 1) # 对角线为1
        cond_proc_matrix[:2, :] = 0 # <unk>, <pad> = 0
        cond_proc_matrix[:, :2] = 0 # # <unk>, <pad> = 0
        if config['TASK'] != 'REC':
            cond_drug_matrix[:2, :] = 0 # <unk>, <pad> = 0
            cond_drug_matrix[:, :2] = 0 # # <unk>, <pad> = 0； # drugrec不能直接进行，因为它前面被padding掉。或者从visit 1开始；DRL TRAIN的时候反正会对pad特殊处理

        # for diag in diag2diag.keys():
        #     if cond_proc_matrix[diag_voc_all[diag]].sum()==0:
        #         print("CCCCCCC", diag)
        # for diag in diag2diag.keys(): # drugrec有空的
        #     if cond_drug_matrix[diag_voc_all[diag]].sum()==0:
        #         print("CCCCCCC", diag)

        # cf data & co_data
        cf_data = diag2diag.keys() # from train set 3762； 只出现了这么多；
        cf_data = [i for i in cf_data if i not in set(rare_disease)] # common disease
        print("CF_data", len(cf_data))
        cf_data = np.array(list(cf_data)) # ['408.1']
        co_data = np.array([diag_id2_name.get(code, 'blank') for code in cf_data]) # 去掉blank
        indices = np.where(co_data != 'blank')[0]
        cf_data = cf_data[indices] # filter2 过滤掉blank
        print("Content_data", len(cf_data))

        cf_data = np.array([diag_voc_all[code] for code in cf_data]) # 转为ID序列
        co_data = cf_data

        return cf_data, co_data, ref_matrix, cond_proc_matrix, cond_drug_matrix, all_diag_name, all_proc_name, all_drug_name

    def __len__(self):
        return len(self.cf_data)

    def __getitem__(self, index):
        cf = self.cf_data[index] # ID
        co = self.co_data[index]

        row = self.ref_matrix[cf] # 找到对应的行

        positive_indices = np.where(row > 0)[0]
        negative_indices = np.where(row == 0)[0]

        if len(positive_indices) > self.ref_k:
            positive_indices = np.random.choice(positive_indices, self.ref_k, replace=False) # , p=row按权重采集
        if len(negative_indices) > self.ref_k:
            negative_indices = np.random.choice(negative_indices, self.ref_k, replace=False)

        # positive_embedding = self.data[positive_indices] # K, D， pad放到后面去，在batch中做。
        # negative_embedding = self.data[negative_indices]

        cond_proc = self.cond_proc_matrix[cf]
        cond_proc_indices = np.where(cond_proc > 0)[0]

        if len(cond_proc_indices) > self.cond_k: # proj太少了
            cond_proc_indices = np.random.choice(cond_proc_indices, self.cond_k, replace=False) # p=cond_proc
        cond_drug = self.cond_drug_matrix[cf]
        cond_drug_indices = np.where(cond_drug > 0)[0]
        if len(cond_drug_indices) > self.cond_k:
            cond_drug_indices = np.random.choice(cond_drug_indices, self.cond_k, replace=False)

        return {"diag_code": cf, "pos_diag": torch.from_numpy(positive_indices), "neg_diag": torch.from_numpy(negative_indices),
                "cond_proc": torch.from_numpy(cond_proc_indices), "cond_drug": torch.from_numpy(cond_drug_indices)}
        # , co, positive_indices, negative_indices, cond_proc_indices, cond_drug_indices



def get_dataloader_drl(dataset, batch_size, shuffle=False, drop_last=False):
    """for second stage"""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn_dict,

    )

    return dataloader
