# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : config.py
# Time       ：3/7/2024 11:29 am
# Author     ：xxxxx
# version    ：python 
# Description：
"""
pcf_config = {
    'Transformer' : {
        'EPOCH': 30,
        'FINE_EPOCH': 50,
        'FINE_LR': 1e-4, # eICU 1e-5; MIMIC 1e-4
        'DIM': 128,
        'HIDDEN':128,
        'LR': 1e-3,
        'BATCH': 16,
        'RNN_LAYERS': 2,
        'DROPOUT': 0.1,
        'WD': 1e-4,
        'MULTI': 0.0,
        'DDI': 0.08,
        'MAXSEQ': 10,
        'MAXCODESEQ': 512,
        'TOPK': 20,
    },
    'Dipole': {
        'EPOCH': 50,
        'FINE_EPOCH': 5,
        'FINE_LR': 1e-6,
        'DIM': 128,
        'HIDDEN': 128,
        'LR': 2e-4,
        'BATCH': 16,  # 1/4
        'RNN_LAYERS': 2,
        'DROPOUT': 0.1,
        'WD': 0,
        'MULTI': 0.0,
        'DDI': 0.08,
        'MAXSEQ': 10,
        'MAXCODESEQ': 512,
        'TOPK': 20,
    },
    'SHAPE' : {
        'EPOCH': 50,
        'FINE_EPOCH': 20,
        'FINE_LR':2e-5,
        'DIM': 128,
        'HIDDEN':128,
        'LR': 1e-3,
        'BATCH': 16,
        'RNN_LAYERS': 2,
        'DROPOUT': 0.1,
        'WD': 0,
        'MULTI': 0.0,
        'DDI': 0.08,
        'MAXSEQ': 10,
        'MAXCODESEQ': 512,
        'TOPK': 20,
    },
    'StratMed': {  # 需要改trainer以保证匹配
        'EPOCH': 50,
        'FINE_EPOCH': 1,
        'FINE_LR': 1e-6,

        'DIM': 128,
        'HIDDEN': 128,
        'LR': 2e-4,
        'BATCH': 16,
        'RNN_LAYERS': 2,
        'DROPOUT': 0.1,
        'WD': 0,
        'MULTI': 0.0,
        'DDI': 0.08,
        'MAXSEQ': 10,
        'MAXCODESEQ': 512,
        'TOPK': 20,
    },
    'HITNet': {
        'EPOCH': 50,
        'FINE_EPOCH': 1,
        'FINE_LR': 1e-6,

        'DIM': 128,
        'HIDDEN': 128,
        'LR': 2e-4,
        'BATCH': 16,
        'RNN_LAYERS': 2,
        'DROPOUT': 0.1,
        'WD': 0,
        'MULTI': 0.0,
        'DDI': 0.08,
        'MAXSEQ': 10,
        'MAXCODESEQ': 512,
        'TOPK': 20,
    },
    'RAREMed': {
        'EPOCH': 50,
        'FINE_EPOCH': 1,
        'FINE_LR': 1e-6,

        'DIM': 128,
        'HIDDEN': 128,
        'LR': 2e-4,
        'BATCH': 1,
        'RNN_LAYERS': 2,
        'DROPOUT': 0.1,
        'WD': 0,
        'MULTI': 0.0,
        'DDI': 0.08,
        'MAXSEQ': 10,
        'MAXCODESEQ': 512,
        'TOPK': 20,
    },
}


pcf_rec_config = {
    'Transformer' : {
        'EPOCH': 30,
        'FINE_EPOCH': 50,
        'FINE_LR': 1e-3, # eICU 2e-4
        'DIM': 128,
        'HIDDEN':128,
        'LR': 2e-4,
        'BATCH': 16,
        'RNN_LAYERS': 2,
        'DROPOUT': 0.1,
        'WD': 0,
        'MULTI': 0.0,
        'DDI': 0.08,
        'MAXSEQ': 10,
        'MAXCODESEQ': 512,
        'TOPK': 20,
    },
    'Dipole': {
        'EPOCH': 50,
        'FINE_EPOCH': 5,
        'FINE_LR': 1e-6,
        'DIM': 128,
        'HIDDEN': 128,
        'LR': 2e-4,
        'BATCH': 16,  # 1/4
        'RNN_LAYERS': 2,
        'DROPOUT': 0.1,
        'WD': 0,
        'MULTI': 0.0,
        'DDI': 0.08,
        'MAXSEQ': 10,
        'MAXCODESEQ': 512,
        'TOPK': 20,
    },
    'COGNet': { # specific to rec
        'EPOCH': 50,
        'FINE_EPOCH': 1,
        'FINE_LR': 1e-6,

        'DIM': 128,
        'HIDDEN': 128,
        'LR': 1e-3,
        'BATCH': 16,
        'RNN_LAYERS': 2,
        'DROPOUT': 0.1,
        'WD': 0,
        'MULTI': 0.0,
        'DDI': 0.08,
        'MAXSEQ': 10,
        'MAXCODESEQ': 512,
        'TOPK': 20,
    },
    'SHAPE': {
        'EPOCH': 50,
        'FINE_EPOCH': 5, # 正常训练
        'FINE_LR': 1e-6,

        'DIM': 128,
        'HIDDEN': 128,
        'LR': 2e-4,
        'BATCH': 4,
        'RNN_LAYERS': 2,
        'DROPOUT': 0.1,
        'WD': 0,
        'MULTI': 0.0,
        'DDI': 0.08,
        'MAXSEQ': 10,
        'MAXCODESEQ': 512,
        'TOPK': 20,
    },
    'VITA': {
        'EPOCH': 50,
        'FINE_EPOCH': 1,
        'FINE_LR': 1e-6,

        'DIM': 128,
        'HIDDEN': 128,
        'LR': 2e-4,
        'BATCH': 16,
        'RNN_LAYERS': 2,
        'DROPOUT': 0.1,
        'WD': 0,
        'MULTI': 0.0,
        'DDI': 0.08,
        'MAXSEQ': 10,
        'MAXCODESEQ': 512,
        'TOPK': 20,
    },
    'HITNet': {
        'EPOCH': 50,
        'FINE_EPOCH': 1,
        'FINE_LR': 1e-6,

        'DIM': 128,
        'HIDDEN': 128,
        'LR': 2e-4,
        'BATCH': 16,
        'RNN_LAYERS': 2,
        'DROPOUT': 0.1,
        'WD': 0,
        'MULTI': 0.0,
        'DDI': 0.08,
        'MAXSEQ': 10,
        'MAXCODESEQ': 512,
        'TOPK': 20,
    },
    'RAREMed': { # 需要改trainer以保证匹配
        'EPOCH': 50,
        'FINE_EPOCH': 1,
        'FINE_LR': 1e-6,

        'DIM': 128,
        'HIDDEN': 128,
        'LR': 2e-4,
        'BATCH': 4,
        'RNN_LAYERS': 1,
        'DROPOUT': 0.1,
        'WD': 0.1,
        'MULTI':0 ,# 0.005,
        'DDI': 0.08,
        'MAXSEQ': 10, # 不然不够长，有的很长
        'MAXCODESEQ': 512,
        'TOPK': 20,
    },
    'StratMed': {  # 需要改trainer以保证匹配
        'EPOCH': 50,
        'FINE_EPOCH': 1,
        'FINE_LR': 1e-6,

        'DIM': 128,
        'HIDDEN': 128,
        'LR': 2e-4,
        'BATCH': 16,
        'RNN_LAYERS': 2,
        'DROPOUT': 0.1,
        'WD': 0,
        'MULTI': 0.0,
        'DDI': 0.08,
        'MAXSEQ': 10,
        'MAXCODESEQ': 512,
        'TOPK': 20,
    },
    'DEPOT': {  # 需要改trainer以保证匹配
        'EPOCH': 50,
        'FINE_EPOCH': 1,
        'FINE_LR': 1e-6,

        'DIM': 128,
        'HIDDEN': 128,
        'LR': 1e-3,
        'BATCH': 16,
        'RNN_LAYERS': 2,
        'DROPOUT': 0.1,
        'WD': 0,
        'MULTI': 0.0,
        'DDI': 0.08,
        'MAXSEQ': 10,
        'MAXCODESEQ': 512,
        'TOPK': 20,
    },
}

class UDCDIAGConfig(): # 不要有drugs
    """DRL config"""
    # data_info parameter
    DEV = False
    MODEL = "UDC"
    TASK = 'DIAG'
    DATASET = 'MIV'
    PCF_MODEL = 'Transformer' # StratMed准备老长时间了。
    FEATURE = ['conditions', 'procedures', 'drugs']
    LABEL = 'labels'
    PCF_CONFIG = pcf_config[PCF_MODEL]
    PLM_MODEL = 'Sap-BERT' #不同的对应不同版本的transofmers(4.28.1) 4.44.2

    ATCLEVEL = 3
    RATIO = 0.6 # train-test split
    THRES = 0.4 # pred threshold
    RARE_THRES = 0.8 # rare disease (threshold 0.5, 0.9)

    # train parameter
    SEED = 528
    USE_CUDA = True
    GPU = '4'
    EPOCH = 50
    DIM = 128
    HIDDEN = 128
    LR = 1e-3

    BATCH = 32
    DROPOUT = 0.1
    WD = 5e-4

    # prepro
    REFK = 20
    CONDK = 10
    N_CODEBOOK = 4
    N_EMBED = 64

    # log
    LOGDIR = '/home/xxxx/UDCHealth/log/ckpt/'



class UDCDRECConfig():
    """DRL config"""
    # data_info parameter
    DEV = False # 是否使用sample子集
    MODEL = "UDC"
    TASK = 'REC'
    DATASET = 'MIII' # MIMIMC-III, MIMIC-IV , eICU, OMOP, PIC, OMIX
    PCF_MODEL = 'Transformer' # Transformer, SHAPE, DEPOT,StratMed,RAREMED | COGNET
    FEATURE = ['conditions', 'procedures', 'drugs'] # drug_HIST两层含义
    LABEL = 'labels'
    PCF_CONFIG = pcf_rec_config[PCF_MODEL]
    PLM_MODEL = 'Sap-BERT'

    ATCLEVEL = 3
    RATIO = 0.6
    THRES = 0.4 # pred threshold
    RARE_THRES = 0.8 # rare disease threshold； 

    # train parameter
    SEED = 528
    USE_CUDA = True
    GPU = '3'
    EPOCH = 50 #
    DIM = 128
    HIDDEN = 256
    LR = 1e-4

    BATCH = 32
    DROPOUT = 0.1
    WD = 5e-4

    # prepro
    REFK = 10
    CONDK = 10
    N_CODEBOOK = 4
    N_EMBED = 64

    # log
    LOGDIR = '/home/xxxx/UDCHealth/log/ckpt/'



# config = vars(UDCDIAGConfig)
# config = {k:v for k,v in config.items() if not k.startswith('__')}

config = vars(UDCDRECConfig)
config = {k:v for k,v in config.items() if not k.startswith('__')}
#
