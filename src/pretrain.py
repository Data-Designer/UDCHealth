# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : pretrain.py
# Time       ：2/8/2024 10:52 am
# Author     ：XXXXX
# version    ：python 
# Description：PCF，DRL这里需要对DRL进行处理
"""
import torch
import math
import numpy as np
from models import underlying_model, UDCHealth
from trainer import Trainer
from pretrain_trainer import PTrainer
from transformers import AutoTokenizer, AutoModel
from loader import DRLDataset, get_dataloader_drl
from tqdm import tqdm
from models import DRL
from utils import save_pickle, load_pickle
# from transformers import BioGptTokenizer, BioGptForCausalLM
from transformers import BioGptTokenizer, BioGptModel


def load_pretrain_pcf(sample_dataset, config, special_input = None, exp_num=''):
    model = underlying_model(config,
                             sample_dataset,
                             feature_keys=config['FEATURE'],
                             label_key=config['LABEL'],
                             special_input=special_input

                             )
    ckpt_path = config['LOGDIR'] + config['TASK'] +  '/' + config['DATASET'] + '-' + config['PCF_MODEL'] + '-' + exp_num + '/best.ckpt',
    print("PCF Load dir, ", ckpt_path)
    state_dict = torch.load(ckpt_path[0], map_location='cuda:' + config['GPU'])
    model.load_state_dict(state_dict, strict=False)
    return model

def evaluate_pcf(model, test_dataloader,config,y_grouped =None, p_grouped=None, exp_num=''):
    if config['TASK'] == 'REC':
        p_grouped = p_grouped
        monitor = 'roc_auc_samples'
        metrics = ['jaccard_samples', 'f1_samples', 'pr_auc_samples', 'roc_auc_samples', 'precision_samples',
                   'recall_samples', 'group_rec']
    elif config['TASK'] == 'DIAG':
        y_grouped = y_grouped
        monitor = 'topk_precision'
        metrics = ['jaccard_samples', 'f1_samples', 'pr_auc_samples', 'roc_auc_samples', 'precision_samples',
                   'recall_samples']+['topk_acc', 'topk_precision']
    else:
        raise ValueError("Task not supported!")
    trainer = Trainer(
        model=model,
        metrics=metrics,  # 换指标
        device='cuda:' + config['GPU'] if config['USE_CUDA'] else 'cpu',
        output_path=config['LOGDIR'] + config['TASK']  + '/',
        exp_name=config['DATASET'] + '-' + config['PCF_MODEL'] +'-' + exp_num,  #
    )

    config = config['PCF_CONFIG']
    scores= trainer.evaluate(test_dataloader, aux_data={'topk':config['TOPK'], 'y_grouped':y_grouped, 'p_grouped':p_grouped})

    for key in scores.keys():
        if key.endswith('grouped'):
            print("{}: {}".format(key, scores[key]))  # 列表
        else:
            print("{}: {:4f}".format(key, scores[key]))  # 浮点数

    # scores = trainer.evaluate(test_dataloader,
    #                           aux_data={'topk': 10, 'y_grouped': y_grouped, 'p_grouped': p_grouped})
    #
    # for key in scores.keys():
    #     if key.endswith('grouped'):
    #         print("{}: {}".format(key, scores[key]))  # 列表
    #     else:
    #         print("{}: {:4f}".format(key, scores[key]))  # 浮点数
    #
    # scores = trainer.evaluate(test_dataloader,
    #                           aux_data={'topk': 20, 'y_grouped': y_grouped, 'p_grouped': p_grouped})
    #
    # for key in scores.keys():
    #     if key.endswith('grouped'):
    #         print("{}: {}".format(key, scores[key]))  # 列表
    #     else:
    #         print("{}: {:4f}".format(key, scores[key]))  # 浮点数
    #
    #
    # scores = trainer.evaluate(test_dataloader,
    #                           aux_data={'topk': 40, 'y_grouped': y_grouped, 'p_grouped': p_grouped})
    #
    # for key in scores.keys():
    #     if key.endswith('grouped'):
    #         print("{}: {}".format(key, scores[key]))  # 列表
    #     else:
    #         print("{}: {:4f}".format(key, scores[key]))  # 浮点数

    return

def load_pretrain_plm(config, exp_num=''):
    path = config['LOGDIR'] + config['TASK'] +  '/' + config['DATASET'] + '-' + config['PCF_MODEL'] + '-'+ exp_num + '/plm.pth'
    print("PLM Load dir, ", path)

    file = load_pickle(path)
    diag_embs_plm, proc_embs_plm, drug_embs_plm = file['diag_embs_plm'], file['proc_embs_plm'], file['drug_embs_plm']
    return diag_embs_plm, proc_embs_plm, drug_embs_plm

def load_pretrain_drl(pcf_model, plm_model, config, exp_num=''):
    diag_embs_pcf = pcf_model.embeddings['conditions'].weight.data
    proc_embs_pcf = pcf_model.embeddings['procedures'].weight.data
    drug_embs_pcf = pcf_model.embeddings['drugs'].weight.data
    diag_embs_plm, proc_embs_plm, drug_embs_plm = plm_model[0], plm_model[1], plm_model[2]
    model = DRL(
        mode='regression',
        pcf_embedding=diag_embs_pcf,
        plm_embedding=diag_embs_plm,
        pcf_proc_cond_embedding=proc_embs_pcf,
        plm_proc_cond_embedding=proc_embs_plm,
        pcf_drug_cond_embedding=drug_embs_pcf,
        plm_drug_cond_embedding=drug_embs_plm,
        config=config
    )
    ckpt_path = config['LOGDIR'] + config['TASK'] +  '/' + config['DATASET'] + '-' + config['PCF_MODEL'] + '-drl' + '-'+exp_num+ '/best.ckpt',

    print("DRL Load dir, ", ckpt_path)
    print("AAAAAAAAAA", config['JOINT'])
    # if not config['JOINT']:
    #     print("We donnot need load pretrain DRL!")
    state_dict = torch.load(ckpt_path[0], map_location='cuda:' + config['GPU'])
    model.load_state_dict(state_dict, strict=False)

    return model

def run_pretrain_pcf(sample_dataset, train_dataloader, test_dataloader, config, y_grouped=None, p_grouped=None, special_input = None, exp_num=''):
    # model definition
    model = underlying_model(config,
                             sample_dataset,
                             feature_keys=config['FEATURE'],
                             label_key=config['LABEL'],
                             special_input=special_input
                             )
    if config['TASK'] == 'REC':
        p_grouped = p_grouped
        monitor = 'roc_auc_samples'
        metrics = ['jaccard_samples', 'f1_samples', 'pr_auc_samples', 'roc_auc_samples', 'precision_samples', 'recall_samples', 'group_rec']
    elif config['TASK'] == 'DIAG':
        y_grouped = y_grouped
        monitor = 'topk_precision'
        metrics = ['jaccard_samples', 'f1_samples', 'pr_auc_samples', 'roc_auc_samples', 'precision_samples',
                   'recall_samples']+['topk_acc', 'topk_precision']
    else:
        raise ValueError("Task not supported!")
    trainer = Trainer(
        model=model,
        metrics=metrics,  # 换指标
        device='cuda:' + config['GPU'] if config['USE_CUDA'] else 'cpu',
        output_path=config['LOGDIR'] + config['TASK'] +  '/',
        exp_name= config['DATASET'] + '-' + config['PCF_MODEL'] + '-' + exp_num,  #
    )

    if config['PCF_MODEL'] == 'RAREMed':
        config = config['PCF_CONFIG']
        trainer.train_prefix(
            train_dataloader=train_dataloader,
            val_dataloader=test_dataloader,  # test_dataloader,
            test_dataloader=test_dataloader,  # 检查，可能有东西没保存
            epochs=config['EPOCH'],
            weight_decay=config['WD'],
            # steps_per_epoch=20,  # 检查
            monitor=monitor,  # roc_auc
            optimizer_params={"lr": config['LR']},
            max_grad_norm=0.1,
            load_best_model_at_last=True,
            aux_data={'topk': config['TOPK'], 'y_grouped': y_grouped, 'p_grouped': p_grouped}
        )
    else:
        print("PCF Has been trained!")
        # pass
        config = config['PCF_CONFIG']
        trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=test_dataloader, # test_dataloader,
            # test_dataloader=test_dataloader, # 检查，可能有东西没保存
            epochs=config['EPOCH'],
            weight_decay = config['WD'],
            # steps_per_epoch=200, # 检查
            monitor= monitor, # roc_auc
            optimizer_params={"lr": config['LR']},
            max_grad_norm=0.1,
            load_best_model_at_last=True,
            aux_data={'topk':config['TOPK'], 'y_grouped':y_grouped, 'p_grouped':p_grouped}
        )
    print("Final Test")
    scores = trainer.evaluate(test_dataloader, aux_data={'topk':config['TOPK'], 'y_grouped':y_grouped, 'p_grouped':p_grouped})
    print(scores)
    return trainer.model


def run_pretrain_plm(model_name, config):
    device = 'cuda:' + config['GPU'] if config['USE_CUDA'] else 'cpu'
    if model_name == 'Sap-BERT':
        tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(device)
    elif model_name == 'BioGPT':
        tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        model = BioGptModel.from_pretrained("microsoft/biogpt").to(device)
        # text = "Replace me by any text you'd like."
        # encoded_input = tokenizer(text, return_tensors='pt')
        # output = model(**encoded_input)
    elif model_name == 'BioMistral':
        tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")# 需要torch 1.6
        model = AutoModel.from_pretrained("BioMistral/BioMistral-7B").to(device)
    elif model_name == 'Clinical-BERT':
        tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        model = AutoModel.from_pretrained("medicalai/ClinicalBERT").to(device)
    else:
        raise ValueError("Model not supported!")
    return model, tokenizer


def get_embedding(model, tokenizer, all_text, plm_model_name):
    """防止有些model需要特殊处理"""
    bs = 16  # batch size during inference
    all_embs = []
    if plm_model_name == 'Sap-BERT':
        for i in tqdm(np.arange(0, len(all_text), bs)):
            try:
                toks = tokenizer.batch_encode_plus(all_text[i:i + bs],
                                                   padding="max_length",
                                                   max_length=25,
                                                   truncation=True,
                                                   return_tensors="pt")
            except:
                updated_list = ['Unknown' if (isinstance(v, float) and math.isnan(v)) or v != v else v for v in
                                all_text[i:i + bs]]

                toks = tokenizer.batch_encode_plus(updated_list,
                                                   padding="max_length",
                                                   max_length=25,
                                                   truncation=True,
                                                   return_tensors="pt")

            toks_cuda = {}
            for k, v in toks.items():
                toks_cuda[k] = v.to(model.device)
            cls_rep = model(**toks_cuda)[0][:, 0, :]  # use CLS representation as the embedding
            all_embs.append(cls_rep.cpu().detach())
    elif plm_model_name in ['BioGPT', 'Clinical-BERT', 'BioMistral']:
        for i in tqdm(np.arange(0, len(all_text), bs)):
            try:
                toks = tokenizer(all_text[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=50)
            except:
                updated_list = ['Unknown' if (isinstance(v, float) and math.isnan(v)) or v != v else v for v in
                                all_text[i:i + bs]]
                toks = tokenizer(updated_list, return_tensors="pt", padding=True, truncation=True)

            toks_cuda = {}
            for k, v in toks.items():
                toks_cuda[k] = v.to(model.device)
            embeddings = model(**toks_cuda).last_hidden_state  # use CLS representation as the embedding
            sentence_embeddings = torch.mean(embeddings, dim=1)
            all_embs.append(sentence_embeddings.cpu().detach())

    all_embs = torch.cat(all_embs, dim=0)
    return all_embs


def run_pretrain_drl(disease, sample_dataset, train_dataset, test_dataloader, pcf_model, config, exp_num=''):
    """这里或许可以换成val_dataloder, 不过有点浪费时间; """
    # pretrain underlying
    pcf_model = pcf_model
    plm_model, tokenizer = run_pretrain_plm(config['PLM_MODEL'], config)
    print("Pretrain PLM done!")

    # dataset preparation
    dataset = DRLDataset(disease, sample_dataset, train_dataset, config)
    diag_text, cond_proc_text, cond_drug_text = dataset.diag_text, dataset.cond_proc_text, dataset.cond_drug_text
    # dataloader
    data_loader = get_dataloader_drl(dataset, batch_size=config['BATCH'], shuffle=True, drop_last=False)
    print("DRL dataloader done!", len(data_loader)) # 4*830 batch

    # embedding preparation
    diag_embs_pcf = pcf_model.embeddings['conditions'].weight.data
    proc_embs_pcf = pcf_model.embeddings['procedures'].weight.data
    if 'drugs' in config['FEATURE']:
        drug_embs_pcf = pcf_model.embeddings['drugs'].weight.data
        drug_embs_plm = get_embedding(plm_model, tokenizer, cond_drug_text, config['PLM_MODEL'])
    else:
        drug_embs_pcf = None
        drug_embs_plm = None
    diag_embs_plm = get_embedding(plm_model, tokenizer, diag_text, config['PLM_MODEL'])
    proc_embs_plm = get_embedding(plm_model, tokenizer, cond_proc_text, config['PLM_MODEL'])
    path = config['LOGDIR'] + config['TASK'] +  '/' + config['DATASET'] + '-' + config['PCF_MODEL'] + '-' + exp_num + '/plm.pth' # 基本上是不同的
    save_pickle({'diag_embs_plm': diag_embs_plm, 'proc_embs_plm': proc_embs_plm, 'drug_embs_plm': drug_embs_plm}, path)

    assert diag_embs_pcf.shape[0] == diag_embs_plm.shape[0]
    print("Shape Examination", diag_embs_pcf.shape, diag_embs_plm.shape) # 16*256 batch


    # DRL model definition
    model = DRL(
        mode='regression',
        pcf_embedding=diag_embs_pcf,
        plm_embedding=diag_embs_plm,
        pcf_proc_cond_embedding=proc_embs_pcf,
        plm_proc_cond_embedding=proc_embs_plm,
        pcf_drug_cond_embedding=drug_embs_pcf,
        plm_drug_cond_embedding=drug_embs_plm,
        config=config
    )

    # if config['TASK'] == 'REC':
        # monitor = 'roc_auc_samples'
        # metrics = ['jaccard_samples', 'f1_samples', 'pr_auc_samples', 'roc_auc_samples', 'precision_samples', 'recall_samples']
    # elif config['TASK'] == 'DIAG':
    metrics = ['rmse', 'mae']
    monitor = 'pcf_rmse'

    # trainer
    trainer = PTrainer(
        model=model,
        metrics=metrics,
        device='cuda:' + config['GPU'] if config['USE_CUDA'] else 'cpu',
        output_path=config['LOGDIR'] + config['TASK'] + '/',
        exp_name=config['DATASET'] + '-' + config['PCF_MODEL'] + '-drl'+'-'+exp_num,
    )

    trainer.train(
        train_dataloader=data_loader,
        val_dataloader=data_loader,  # 保证test_loader
        test_dataloader=data_loader,  #
        epochs=config['EPOCH'],
        weight_decay=config['WD'],
        # steps_per_epoch=200, # 检查
        monitor=monitor,  # 换成mae
        monitor_criterion='min',
        optimizer_params={"lr": config['LR']},
        max_grad_norm=0.1,
        load_best_model_at_last=True # 必须要load
    )

    drl_model = trainer.model


    return plm_model, drl_model



def aug_inference(dataset, pcf_model, plm_model, drl_model, train_dataloader, test_dataloader, config, y_grouped=None, p_grouped=None, special_input=None, tuning=False, exp_num='',**kwargs):
    model = UDCHealth(
        dataset,
        pcf_model,
        plm_model,
        drl_model,
        feature_keys=config['FEATURE'],
        label_key=config["LABEL"],
        mode="multilabel",

        # joint train
        train_dataset=special_input,
        config=config,
    )
    if config['TASK'] == 'REC':
        metrics = ['jaccard_samples', 'f1_samples', 'pr_auc_samples', 'roc_auc_samples', 'precision_samples', 'recall_samples', 'group_rec']
        monitor = 'roc_auc_samples'

    elif config['TASK'] == 'DIAG':
        y_grouped = y_grouped
        monitor = 'topk_precision'
        metrics = ['jaccard_samples', 'f1_samples', 'pr_auc_samples', 'roc_auc_samples', 'precision_samples',
                   'recall_samples']+['topk_acc', 'topk_precision']

    trainer = Trainer(
        model=model,
        metrics=metrics,  # 换指标
        device='cuda:' + config['GPU'] if config['USE_CUDA'] else 'cpu',
        output_path=config['LOGDIR'] + config['TASK'] + '/',
        exp_name=config['DATASET'] + '-' + config['PCF_MODEL'] + '-udc' + '-' + exp_num,

    )
    config = config['PCF_CONFIG']

    print("=====no tuning:")
    scores = trainer.evaluate(test_dataloader,
                     aux_data={'topk':config['TOPK'], 'y_grouped':y_grouped,'p_grouped':p_grouped}
                     )

    for key in scores.keys():
        if key.endswith('grouped'):
            print("{}: {}".format(key, scores[key]))  # 列表
        else:
            print("{}: {:4f}".format(key, scores[key]))  # 浮点数


    if tuning:
        print("=====tuning start:")
        #
        for param in model.drl_model.parameters(): # joint
            param.requires_grad = False
        # for param in model.pcf_model.parameters(): # 只更新rec
        #     param.requires_grad = False
        # for param in model.pcf_model.isab.parameters(): # 只有交互需要重新捕获
        #     param.requires_grad = True
        # for param in model.drl_model.task_aware.parameters():
        #     param.requires_grad = True

        # for param in model.pcf_model.rec_layer.fina_proj.parameters(): # 只更新rec
        #     param.requires_grad = True
        # for param in model.pcf_model.embeddings['conditions'].parameters(): # 这个倒是真的。 joint
        #     param.requires_grad = False # 交互方式, 这个原本的语义空间感觉一定不能动，一旦动了，就不是原来的语义空间了，那么前者的对齐将毫无用处。

        trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=test_dataloader, # test_dataloader,
            # test_dataloader=test_dataloader, # 检查，可能有东西没保存
            epochs=config['FINE_EPOCH'],
            weight_decay = config['WD'],
            # steps_per_epoch=200, # 检查
            monitor= monitor, # roc_auc
            optimizer_params={"lr": config['FINE_LR']},
            max_grad_norm=0.1,
            load_best_model_at_last=True,
            aux_data={'topk':config['TOPK'], 'y_grouped':y_grouped, 'p_grouped':p_grouped}
        )
    print("===========================Tuning Done!==========================")
    scores = trainer.evaluate(test_dataloader,
                              aux_data={'topk': config['TOPK'], 'y_grouped': y_grouped, 'p_grouped': p_grouped}
                              )
    print(scores)

    return model





if __name__ == '__main__':
    pass
