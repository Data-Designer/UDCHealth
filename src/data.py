# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : data.py
# Time       ：8/3/2024 9:13 am
# Author     ：XXXXX
# version    ：python 
# Description：dataset generator, 设定task
"""
from pyhealth.data import Patient, Visit
from config import config
from pyhealth.datasets import SampleEHRDataset, SampleBaseDataset
from pyhealth.datasets.utils import list_nested_levels
from typing import Dict, List


def re_generate_dataset(samples, seed):
    sample_dataset = SampleEHRDataset(  # 这个贼耗时
        samples,
        dataset_name=config['DATASET'],
        task_name=config['TASK'],
    )

    #  split & save dataset
    train_dataset,_, test_dataset = split_by_patient(
        sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
        train_ratio=1.0,  # Train test split
        seed=seed,
    )
    print("Regerenate dataset done!")
    return train_dataset, test_dataset



def convert_dataset(samples, dataset_name=None, task_name=None):
    """避免繁琐的处理"""
    return SampleEHRDataset(
                    samples,
                    dataset_name=dataset_name,
                    task_name=task_name,
                    # all=all_samples,
                )    # return SampleEHRDatasetSIMPLE(
    #                 samples,
    #                 dataset_name=dataset_name,
    #                 task_name=task_name,
    #                 all=all_samples,
    #             )


class SampleEHRDatasetSIMPLE(SampleBaseDataset):
    def __init__(self, samples: List[str], code_vocs=None, dataset_name="", task_name="", all=False):
        super().__init__(samples, dataset_name, task_name)
        self.samples = samples
        if all:
            self.input_info: Dict = self._validate() # 别的不需要valid，大大减少时间


    @property
    def available_keys(self) -> List[str]:
        """Returns a list of available keys for the dataset.

        Returns:
            List of available keys.
        """
        keys = self.samples[0].keys()
        return list(keys)

    def _validate(self) -> Dict:
        """ 1. Check if all samples are of type dict. """
        keys = self.samples[0].keys()

        """
        4. For each key, check if it is either:
            - a single value
            - a single vector
            - a list of codes
            - a list of vectors
            - a list of list of codes
            - a list of list of vectors
        Note that a value is either float, int, or str; a vector is a list of float 
        or int; and a code is str.
        """
        # record input information for each key
        input_info = {}
        for key in keys:
            """
            4.1. Check nested list level: all samples should either all be
            - a single value (level=0)
            - a single vector (level=1)
            - a list of codes (level=1)
            - a list of vectors (level=2)
            - a list of list of codes (level=2)
            - a list of list of vectors (level=3)
            """
            levels = set([list_nested_levels(s[key]) for s in self.samples[:5]]) # 只取前5个判断足够

            level = levels.pop()[0]

            # flatten the list
            if level == 0:
                flattened_values = [s[key] for s in self.samples]
            elif level == 1:
                flattened_values = [i for s in self.samples for i in s[key]]
            elif level == 2:
                flattened_values = [j for s in self.samples for i in s[key] for j in i]
            else:
                flattened_values = [
                    k for s in self.samples for i in s[key] for j in i for k in j
                ]

            """
            4.2. Check type: the basic type of each element should be float, 
            int, or str.
            """
            types = set([type(v) for v in flattened_values[:5]]) # 只取前5个判断足够
            type_ = types.pop()
            """
            4.3. Combined level and type check.
            """
            if level == 0:
                # a single value
                input_info[key] = {"type": type_, "dim": 0}
            elif level == 1:
                # a single vector or a list of codes
                if type_ in [float, int]:
                    # a single vector
                    lens = set([len(s[key]) for s in self.samples])
                    input_info[key] = {"type": type_, "dim": 1, "len": lens.pop()}
                else:
                    # a list of codes
                    # note that dim is different from level here
                    input_info[key] = {"type": type_, "dim": 2}
            elif level == 2:
                # a list of vectors or a list of list of codes
                if type_ in [float, int]:
                    lens = set([len(i) for s in self.samples for i in s[key]])
                    input_info[key] = {"type": type_, "dim": 2, "len": lens.pop()}
                else:
                    # a list of list of codes
                    # note that dim is different from level here
                    input_info[key] = {"type": type_, "dim": 3}
            else:
                # a list of list of vectors
                lens = set([len(j) for s in self.samples for i in s[key] for j in i])
                input_info[key] = {"type": type_, "dim": 3, "len": lens.pop()}

        return input_info

    def __len__(self):
        return len(self.samples)



def exam_warm_cold(conditions):
    """标注是否是warm还是cold,返回0-1"""
    return conditions

def drug_recommendation_mimic3_fn_wc(patient: Patient):
    """
    处理的是一个patient的数据
    Examples:
        {
            'visit_id': '174162',
            'patient_id': '107',
            'conditions': [['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']],
            'procedures': [['4443', '4513', '3995']],
            'drugs_hist': [[]],
            'drugs': ['0000', '0033', '5817', '0057', '0090', '0053', '0', '0012', '6332', '1001', '6155', '1001', '6332', '0033', '5539', '6332', '5967', '0033', '0040', '5967', '5846', '0016', '5846', '5107', '5551', '6808', '5107', '0090', '5107', '5416', '0033', '1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373'],
        }
    """
    samples = []
    for i in range(len(patient)): # visit 次数
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        # ATC 3 level
        drugs = [drug[:config['ATCLEVEL']+1] for drug in drugs]

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_hist": drugs,
                "conditions_wc": exam_warm_cold(conditions),
                "labels": drugs,
            }
        )

    # exclude: patients with less than 2 visit
    if len(samples) < 1: # [{visit 1},{visit 2}]
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]] # 这样visit对应的drug labels不用改;这里最少1个也行
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["conditions_wc"] = [samples[0]["conditions_wc"]]

    for i in range(1, len(samples)): # 第二次，到第N次，一个patient创建一个samples数据，迭代创建
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["conditions_wc"] = samples[i - 1]["conditions_wc"] + [
            samples[i]["conditions_wc"]
        ]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = ['<pad>','<pad>','<pad>'] # 去掉target
        # 时序对齐的padding
        samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
    return samples

def disease_prediction_mimic3_fn_wc(patient: Patient):
    """
    处理的是一个patient的数据
    Examples:
        {
            'visit_id': '174162',
            'patient_id': '107',
            'conditions': [['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']],
            'procedures': [['4443', '4513', '3995']],
            'drugs_hist': [[]],
            'drugs': ['0000', '0033', '5817', '0057', '0090', '0053', '0', '0012', '6332', '1001', '6155', '1001', '6332', '0033', '5539', '6332', '5967', '0033', '0040', '5967', '5846', '0016', '5846', '5107', '5551', '6808', '5107', '0090', '5107', '5416', '0033', '1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373'],
        }
    """
    samples = []
    for i in range(len(patient)-1): # visit 次数
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # print(visit)
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        next_conditions = next_visit.get_code_list(table="DIAGNOSES_ICD")

        # ATC 3 level 'A04D'
        drugs = [drug[:config['ATCLEVEL']+1] for drug in drugs]

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) * len(next_conditions) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_hist": drugs,
                "conditions_wc": exam_warm_cold(conditions),
                "labels": next_conditions,
            }
        )

    # exclude: patients with less than 1 visit
    if len(samples) < 1: # [{visit 1},{visit 2}], 有1的话，其实本身就至少有2次visit
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]] # 这样visit对应的drug labels不用改;这里最少1个也行
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["conditions_wc"] = [samples[0]["conditions_wc"]]

    for i in range(1, len(samples)): #
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["conditions_wc"] = samples[i - 1]["conditions_wc"] + [
            samples[i]["conditions_wc"]
        ]

    # # remove the target drug from the history，disease prediction不需要
    # for i in range(len(samples)): # 都是最后一位
    #     samples[i]["drugs_hist"][i] = ['<pad>','<pad>','<pad>'] # 去掉target
    #     # 时序对齐的padding
    #     samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
    return samples

def drug_recommendation_mimic4_fn_wc(patient: Patient):
    """Processes a single patient for the drug recommendation task.
    Examples:
        {
            'visit_id': '174162',
            'patient_id': '107',
            'conditions': [['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']],
            'procedures': [['4443', '4513', '3995']],
            'drugs_hist': [[]],
            'drugs': ['0000', '0033', '5817', '0057', '0090', '0053', '0', '0012', '6332', '1001', '6155', '1001', '6332', '0033', '5539', '6332', '5967', '0033', '0040', '5967', '5846', '0016', '5846', '5107', '5551', '6808', '5107', '0090', '5107', '5416', '0033', '1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373'],
        }
    """
    samples = []
    for i in range(len(patient)): # visit 次数
        visit: Visit = patient[i]
        # print(visit)
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")

        # ATC 3 level
        drugs = [drug[:config['ATCLEVEL']+1] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_hist": drugs,
                "conditions_wc": exam_warm_cold(conditions),
                "labels": drugs,
            }
        )

    # print('sample', samples)
    # exclude: patients with less than 2 visit
    if len(samples) < 2: # [{},{}]; 这里graphcare动了手脚
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["conditions_wc"] = [samples[0]["conditions_wc"]]

    for i in range(1, len(samples)): # 第二次，到第N次，一个patient创建一个samples数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["conditions_wc"] = samples[i - 1]["conditions_wc"] + [
            samples[i]["conditions_wc"]
        ]


    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = ['<pad>','<pad>','<pad>'] # 去掉target；其实变成unk比较合理
        # 时序对齐的padding
        samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
    return samples




def disease_prediction_mimic4_fn_wc(patient: Patient):
    """Processes a single patient for the drug recommendation task.
    Examples:
        {
            'visit_id': '174162',
            'patient_id': '107',
            'conditions': [['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']],
            'procedures': [['4443', '4513', '3995']],
            'drugs_hist': [[]],
            'drugs': ['0000', '0033', '5817', '0057', '0090', '0053', '0', '0012', '6332', '1001', '6155', '1001', '6332', '0033', '5539', '6332', '5967', '0033', '0040', '5967', '5846', '0016', '5846', '5107', '5551', '6808', '5107', '0090', '5107', '5416', '0033', '1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373'],
        }
    """
    samples = []
    for i in range(len(patient)-1): # visit 次数
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # print(visit)
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        next_conditions = next_visit.get_code_list(table="diagnoses_icd")


        # ATC 3 level
        drugs = [drug[:config['ATCLEVEL']+1] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) * len(next_conditions) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_hist": drugs,
                "conditions_wc": exam_warm_cold(conditions),
                "labels": next_conditions,
            }
        )

    # print('sample', samples)
    # exclude: patients with less than 2 visit
    if len(samples) < 2: # [{},{}]; 这里graphcare动了手脚， 至少有三次
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]] # 这里的drugs_hist本质上就是drug
    samples[0]["conditions_wc"] = [samples[0]["conditions_wc"]]

    for i in range(1, len(samples)): # 第二次，到第N次，一个patient创建一个samples数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["conditions_wc"] = samples[i - 1]["conditions_wc"] + [
            samples[i]["conditions_wc"]
        ]


    # # remove the target drug from the history；
    # for i in range(len(samples)):
    #     samples[i]["drugs_hist"][i] = ['<pad>','<pad>','<pad>'] # 去掉target；其实变成unk比较合理
    #     # 时序对齐的padding
    #     samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
    return samples




# tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
# model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to('cuda:6') # 注意query nn在一个频道
# index_name_dict, all_emb = (load_pickle(config['KG_DATADIR'] + config['DATASET'] + '/ehr_name_map.pkl'),
#                                  load_pickle(config['KG_DATADIR'] + config['DATASET'] + '/entity_emb.pkl'))
# index_name, part = np.array(list(index_name_dict.keys())), round(all_emb.shape[0]//500) # 这玩意巨耗时间
# faiss_index = faiss.IndexFlatL2(768)  # 使用L2距离
# faiss_index.add(all_emb)  # 向索引中添加数据库向量


############eICU==感觉oop也能弄。
def drug_recommendation_eicu_fn_wc(patient: Patient):
    """Processes a single patient for the drug recommendation task.
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_hist": drugs,
                "conditions_wc": exam_warm_cold(conditions),  # 很奇怪，有的时候对不齐；CM和PROC交集其实很少
                "labels": drugs,
            }
        )

    # exclude: patients with less than 2 visit
    if len(samples) < 1:
        return []

    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["conditions_wc"] = [samples[0]["conditions_wc"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]

        samples[i]["conditions_wc"] = samples[i - 1]["conditions_wc"] + [
            samples[i]["conditions_wc"]
        ]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = ['<pad>', '<pad>', '<pad>']  # 去掉target
        # 时序对齐的padding
        samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
    return samples


def disease_prediction_eicu_fn_wc(patient: Patient):
    """Processes a single patient for the drug recommendation task.
    """
    samples = []
    for i in range(len(patient)-1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")
        next_conditions = next_visit.get_code_list(table="diagnosis")

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) * len(next_conditions) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_hist": drugs,
                "conditions_wc": exam_warm_cold(conditions),  # 很奇怪，有的时候对不齐；CM和PROC交集其实很少
                "labels": next_conditions,
            }
        )

    # exclude: patients with less than 2 visit
    if len(samples) < 1:
        return []

    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["conditions_wc"] = [samples[0]["conditions_wc"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]

        samples[i]["conditions_wc"] = samples[i - 1]["conditions_wc"] + [
            samples[i]["conditions_wc"]
        ]

    # # remove the target drug from the history
    # for i in range(len(samples)):
    #     samples[i]["drugs_hist"][i] = ['<pad>', '<pad>', '<pad>']  # 去掉target
    #     # 时序对齐的padding
    #     samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
    return samples



def disease_prediction_pic_fn_wc(patient: Patient):
    """Processes a single patient for the drug recommendation task.
    """
    samples = []
    for i in range(len(patient)-1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        next_conditions = next_visit.get_code_list(table="DIAGNOSES_ICD")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(drugs)* len(next_conditions) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": conditions,
                "drugs": drugs,
                "drugs_hist": drugs,
                "conditions_wc": exam_warm_cold(conditions),  # 很奇怪，有的时候对不齐；CM和PROC交集其实很少
                "labels": next_conditions,
            }
        )

    # exclude: patients with less than 2 visit
    if len(samples) < 1:
        return []

    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["conditions_wc"] = [samples[0]["conditions_wc"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]

        samples[i]["conditions_wc"] = samples[i - 1]["conditions_wc"] + [
            samples[i]["conditions_wc"]
        ]

    # remove the target drug from the history
    # for i in range(len(samples)):
    #     samples[i]["drugs_hist"][i] = ['<pad>', '<pad>', '<pad>']  # 去掉target
    #     # 时序对齐的padding
    #     samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
    return samples


def drug_recommendation_pic_fn_wc(patient: Patient):
    """
    处理的是一个patient的数据
    Examples:
        {
            'visit_id': '174162',
            'patient_id': '107',
            'conditions': [['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']],
            'procedures': [['4443', '4513', '3995']],
            'drugs_hist': [[]],
            'drugs': ['0000', '0033', '5817', '0057', '0090', '0053', '0', '0012', '6332', '1001', '6155', '1001', '6332', '0033', '5539', '6332', '5967', '0033', '0040', '5967', '5846', '0016', '5846', '5107', '5551', '6808', '5107', '0090', '5107', '5416', '0033', '1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373'],
        }
    """
    samples = []
    for i in range(len(patient)): # visit 次数
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        # procedures = visit.get_code_list(table="DIAGNOSES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": conditions,
                "drugs": drugs,
                "drugs_hist": drugs,
                "conditions_wc": exam_warm_cold(conditions),
                "labels": drugs,
            }
        )

    # exclude: patients with less than 2 visit
    if len(samples) < 1: # [{visit 1},{visit 2}]
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]] # 这样visit对应的drug labels不用改;这里最少1个也行
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["conditions_wc"] = [samples[0]["conditions_wc"]]

    for i in range(1, len(samples)): # 第二次，到第N次，一个patient创建一个samples数据，迭代创建
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["conditions_wc"] = samples[i - 1]["conditions_wc"] + [
            samples[i]["conditions_wc"]
        ]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = ['<pad>','<pad>','<pad>'] # 去掉target
        # 时序对齐的padding
        samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
    return samples



def drug_recommendation_omop_fn_wc(patient: Patient):
    """
    处理的是一个patient的数据
    Examples:
        {
            'visit_id': '174162',
            'patient_id': '107',
            'conditions': [['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']],
            'procedures': [['4443', '4513', '3995']],
            'drugs_hist': [[]],
            'drugs': ['0000', '0033', '5817', '0057', '0090', '0053', '0', '0012', '6332', '1001', '6155', '1001', '6332', '0033', '5539', '6332', '5967', '0033', '0040', '5967', '5846', '0016', '5846', '5107', '5551', '6808', '5107', '0090', '5107', '5416', '0033', '1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373'],
        }
    """
    samples = []
    for i in range(len(patient)): # visit 次数
        visit: Visit = patient[i]
        # conditions = visit.get_code_list(table="condition_occurrence")
        procedures = visit.get_code_list(table="procedure_occurrence")
        drugs = visit.get_code_list(table="drug_exposure")

        # ATC 3 level
        # drugs = [drug[:config['ATCLEVEL']+1] for drug in drugs]

        # exclude: visits without condition, procedure, or drug code
        if len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": procedures,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_hist": drugs,
                "conditions_wc": exam_warm_cold(procedures),
                "labels": drugs,
            }
        )

    # exclude: patients with less than 2 visit
    if len(samples) < 1: # [{visit 1},{visit 2}]
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]] # 这样visit对应的drug labels不用改;这里最少1个也行
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["conditions_wc"] = [samples[0]["conditions_wc"]]

    for i in range(1, len(samples)): # 第二次，到第N次，一个patient创建一个samples数据，迭代创建
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["conditions_wc"] = samples[i - 1]["conditions_wc"] + [
            samples[i]["conditions_wc"]
        ]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = ['<pad>','<pad>','<pad>'] # 去掉target
        # 时序对齐的padding
        samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
    return samples


def disease_prediction_omop_fn_wc(patient: Patient):
    """Processes a single patient for the drug recommendation task.
    Examples:
        {
            'visit_id': '174162',
            'patient_id': '107',
            'conditions': [['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']],
            'procedures': [['4443', '4513', '3995']],
            'drugs_hist': [[]],
            'drugs': ['0000', '0033', '5817', '0057', '0090', '0053', '0', '0012', '6332', '1001', '6155', '1001', '6332', '0033', '5539', '6332', '5967', '0033', '0040', '5967', '5846', '0016', '5846', '5107', '5551', '6808', '5107', '0090', '5107', '5416', '0033', '1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373'],
        }
    """
    samples = []
    for i in range(len(patient)-1): # visit 次数
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # print(visit)
        # conditions = visit.get_code_list(table="condition_occurrence")
        procedures = visit.get_code_list(table="procedure_occurrence")
        drugs = visit.get_code_list(table="drug_exposure")
        next_conditions = next_visit.get_code_list(table="procedure_occurrence")


        # ATC 3 level
        # exclude: visits without condition, procedure, or drug code
        if len(procedures) * len(drugs) * len(next_conditions) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": procedures,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_hist": drugs,
                "conditions_wc": exam_warm_cold(procedures),
                "labels": next_conditions,
            }
        )

    # print('sample', samples)
    # exclude: patients with less than 2 visit
    if len(samples) < 1: # [{},{}]; 这里graphcare动了手脚， 至少有三次
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]] # 这里的drugs_hist本质上就是drug
    samples[0]["conditions_wc"] = [samples[0]["conditions_wc"]]

    for i in range(1, len(samples)): # 第二次，到第N次，一个patient创建一个samples数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["conditions_wc"] = samples[i - 1]["conditions_wc"] + [
            samples[i]["conditions_wc"]
        ]


    # # remove the target drug from the history；
    # for i in range(len(samples)):
    #     samples[i]["drugs_hist"][i] = ['<pad>','<pad>','<pad>'] # 去掉target；其实变成unk比较合理
    #     # 时序对齐的padding
    #     samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
    return samples



def drug_recommendation_omix_fn_wc(patient: Patient):
    """
    处理的是一个patient的数据, procedure感觉没啥用啊；
    Examples:
        {
            'visit_id': '174162',
            'patient_id': '107',
            'conditions': [['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']],
            'procedures': [['4443', '4513', '3995']],
            'drugs_hist': [[]],
            'drugs': ['0000', '0033', '5817', '0057', '0090', '0053', '0', '0012', '6332', '1001', '6155', '1001', '6332', '0033', '5539', '6332', '5967', '0033', '0040', '5967', '5846', '0016', '5846', '5107', '5551', '6808', '5107', '0090', '5107', '5416', '0033', '1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373'],
        }
    """
    samples = []
    for i in range(len(patient)): # visit 次数
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="Diagnosis")
        procedures = visit.get_code_list(table="Lab")
        drugs = visit.get_code_list(table="Medication")

        # ATC 3 level
        # drugs = [drug[:config['ATCLEVEL']+1] for drug in drugs]

        # exclude: visits without condition, procedure, or drug code
        if len(conditions)* len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_hist": drugs,
                "conditions_wc": exam_warm_cold(conditions),
                "labels": drugs,
            }
        )

    # exclude: patients with less than 2 visit
    if len(samples) < 1: # [{visit 1},{visit 2}]
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]] # 这样visit对应的drug labels不用改;这里最少1个也行
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["conditions_wc"] = [samples[0]["conditions_wc"]]

    for i in range(1, len(samples)): # 第二次，到第N次，一个patient创建一个samples数据，迭代创建
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["conditions_wc"] = samples[i - 1]["conditions_wc"] + [
            samples[i]["conditions_wc"]
        ]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = ['<pad>','<pad>','<pad>'] # 去掉target
        # 时序对齐的padding
        samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
    return samples



def disease_prediction_omix_fn_wc(patient: Patient):
    """Processes a single patient for the drug recommendation task.
    Examples:
        {
            'visit_id': '174162',
            'patient_id': '107',
            'conditions': [['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']],
            'procedures': [['4443', '4513', '3995']],
            'drugs_hist': [[]],
            'drugs': ['0000', '0033', '5817', '0057', '0090', '0053', '0', '0012', '6332', '1001', '6155', '1001', '6332', '0033', '5539', '6332', '5967', '0033', '0040', '5967', '5846', '0016', '5846', '5107', '5551', '6808', '5107', '0090', '5107', '5416', '0033', '1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373'],
        }
    """
    samples = []
    for i in range(len(patient)-1): # visit 次数
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # print(visit)
        conditions = visit.get_code_list(table="Diagnosis")
        procedures = visit.get_code_list(table="Lab")
        drugs = visit.get_code_list(table="Medication")
        next_conditions = next_visit.get_code_list(table="Diagnosis")


        # ATC 3 level
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) * len(next_conditions) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_hist": drugs,
                "conditions_wc": exam_warm_cold(conditions),
                "labels": next_conditions,
            }
        )

    # print('sample', samples)
    # exclude: patients with less than 2 visit
    if len(samples) < 2: # [{},{}]; 这里graphcare动了手脚， 至少有三次
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]] # 这里的drugs_hist本质上就是drug
    samples[0]["conditions_wc"] = [samples[0]["conditions_wc"]]

    for i in range(1, len(samples)): # 第二次，到第N次，一个patient创建一个samples数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["conditions_wc"] = samples[i - 1]["conditions_wc"] + [
            samples[i]["conditions_wc"]
        ]


    # # remove the target drug from the history；
    # for i in range(len(samples)):
    #     samples[i]["drugs_hist"][i] = ['<pad>','<pad>','<pad>'] # 去掉target；其实变成unk比较合理
    #     # 时序对齐的padding
    #     samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
    return samples









if __name__ == '__main__':
    from pyhealth.datasets import MIMIC3Dataset
    from pyhealth.datasets import split_by_patient
    base_dataset = MIMIC3Dataset(
        root="/home/xxxxx/KnowHealth/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        dev=True,
        refresh_cache=False,
    )
    base_dataset.stat()

    # STEP 2: set task
    sample_dataset = base_dataset.set_task(drug_recommendation_mimic3_fn_wc)
    sample_dataset.stat()
    print(sample_dataset.samples[1])
    from pyhealth.tokenizer import Tokenizer
    id_tokenizer = Tokenizer(
        sample_dataset.get_all_tokens(key='patient_id'),
    )
    id_index = id_tokenizer.convert_tokens_to_indices(['102', '103'])
    print(id_index)


