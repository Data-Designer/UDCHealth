# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : models.py
# Time       ：3/7/2024 11:29 am
# Author     ：XXXXXXX
# version    ：python 
# Description：
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from layer import _get_activation_fn, _get_clones, MLP, Quantize
from loss import RqVaeLoss
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Tuple, Optional, Union
from basemodel import Transformer, COGNet, SHAPE, StratMed, VITA, HITNet, RAREMed, DEPOT, Dipole
from pyhealth.models import BaseModel
# from pyhealth.models.utils import get_last_visit
from utils import get_last_visit, get_tokenizers, get_name_map, get_last_visit_sample, off_diagonal
from layer import Encoder, EncoderLayer, MLP
from collections import defaultdict




##### Rec Model
def underlying_model(config, sample_dataset, feature_keys, label_key, mode="multilabel", special_input=None):
    choice = config['PCF_MODEL']
    pcf_config = config['PCF_CONFIG']
    if choice == "Transformer":
        return Transformer(
            sample_dataset,
            feature_keys = feature_keys,
            label_key = label_key,
            mode = mode,
            # hyper related
            dropout = pcf_config['DROPOUT'],
            num_rnn_layers = pcf_config['RNN_LAYERS'],
            embedding_dim = pcf_config['DIM'],
            hidden_dim = pcf_config['HIDDEN'],
        )
    elif choice == "Dipole":
        return Dipole(
            sample_dataset,
            feature_keys = feature_keys,
            label_key = label_key,
            mode = mode,
            # hyper related
            dropout = pcf_config['DROPOUT'],
            num_rnn_layers = pcf_config['RNN_LAYERS'],
            embedding_dim = pcf_config['DIM'],
            hidden_dim = pcf_config['HIDDEN'],
        )
    elif choice == "COGNet":
        return COGNet(
            sample_dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            # hyper related
            dropout=pcf_config['DROPOUT'],
            num_rnn_layers=pcf_config['RNN_LAYERS'],
            embedding_dim=pcf_config['DIM'],
            hidden_dim=pcf_config['HIDDEN'],
        )
    elif choice == "SHAPE":
        return SHAPE(
            sample_dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            # hyper related
            dropout=pcf_config['DROPOUT'],
            num_rnn_layers=pcf_config['RNN_LAYERS'],
            embedding_dim=pcf_config['DIM'],
            hidden_dim=pcf_config['HIDDEN'],
        )
    elif choice == 'StratMed': # 暂时实现不来；
        return StratMed(
            sample_dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            # hyper related
            dropout=pcf_config['DROPOUT'],
            num_rnn_layers=pcf_config['RNN_LAYERS'],
            embedding_dim=pcf_config['DIM'],
            hidden_dim=pcf_config['HIDDEN'],
            train_dataset=special_input,
        )
    elif choice == 'VITA':
        return VITA(
            sample_dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            # hyper related
            dropout=pcf_config['DROPOUT'],
            num_rnn_layers=pcf_config['RNN_LAYERS'],
            embedding_dim=pcf_config['DIM'],
            hidden_dim=pcf_config['HIDDEN'],
        )
    elif choice == 'HITNet':
        return HITNet(
            sample_dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            # hyper related
            dropout=pcf_config['DROPOUT'],
            num_rnn_layers=pcf_config['RNN_LAYERS'],
            embedding_dim=pcf_config['DIM'],
            hidden_dim=pcf_config['HIDDEN'],
        )
    elif choice == 'RAREMed':
        return RAREMed(
            sample_dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            # hyper related
            dropout=pcf_config['DROPOUT'],
            num_rnn_layers=pcf_config['RNN_LAYERS'],
            embedding_dim=pcf_config['DIM'],
            hidden_dim=pcf_config['HIDDEN'],
        )
    elif choice == 'DEPOT': # 只能够搞med
        return DEPOT(
            sample_dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            # hyper related
            dropout=pcf_config['DROPOUT'],
            num_rnn_layers=pcf_config['RNN_LAYERS'],
            embedding_dim=pcf_config['DIM'],
            hidden_dim=pcf_config['HIDDEN'],
        )
    elif choice == 'Dipole': # 只能够搞med
        return Dipole(
            sample_dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            # hyper related
            dropout=pcf_config['DROPOUT'],
            num_rnn_layers=pcf_config['RNN_LAYERS'],
            embedding_dim=pcf_config['DIM'],
            hidden_dim=pcf_config['HIDDEN'],
        )

    else:
        raise ValueError("Invalid Model choice")



###### Pretrain PCF, get PLM, Encoder；有多个codebook
class PCF_Encoder(nn.Module):
    """Pretrain PCF, provide options, 其实可以直接训练。
    这里暂时采取RQ-VAE的策略; 后期可以考虑换成多个codebook+ KL散度
    """
    def __init__(self, emb_dim, hidden_dim):
        super(PCF_Encoder, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.Dropout(0.1)) #nn.Linear(emb_dim, hidden_dim)

    def forward(self, pcf_input):
        res = self.proj(pcf_input) # 中间表征
        return res

class PLM_Encoder(nn.Module):
    """Pretrain PLM, provide options, 其实可以直接训练。
    这里暂时采取RQ-VAE的策略; 后期可以考虑换成多个codebook+ KL散度
    """
    def __init__(self, emb_dim, hidden_dim): # d_model和hidden_dim 一样
        super(PLM_Encoder, self).__init__()

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        # d_model = self.hidden_dim

        # semantic extract
        # self.encoder_layer = EncoderLayer(d_model=d_model, nhead=2)
        # self.encoder = Encoder(self.encoder_layer, num_layers=2)
        # self.affine_matrix = nn.Linear(emb_dim, d_model)
        # self.encoder = nn.Sequential(nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU())

        self.proj = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.Dropout(0.1))
        # self.proj = MLP(emb_dim, hidden_dim, hidden_dim)


    def forward(self, plm_input):
        # print("AAAAAA", plm_input.shape)
        # feature = self.affine_matrix(plm_input)
        # feature = self.encoder(feature)
        res = self.proj(plm_input) # 中间表征
        return res


class CrossQuant(nn.Module):
    def __init__(self, n_embeddings, n_codebook, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5, nhead=4, config=None):
        super(CrossQuant, self).__init__() # 0.25
        self.commitment_cost = commitment_cost
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon

        # init_bound = 1 / 400
        # embedding = torch.Tensor(n_embeddings, embedding_dim) # codebook
        self.n_codebook = n_codebook
        self.attention_pcf = nn.ModuleList(modules=[
            nn.MultiheadAttention(embedding_dim, num_heads=nhead, batch_first=True)
            for _ in range(n_codebook)
        ])
        self.attention_plm = nn.ModuleList(modules=[
            nn.MultiheadAttention(embedding_dim, num_heads=nhead, batch_first=True)
            for _ in range(n_codebook)
        ])

        #
        device = 'cuda:'+config['GPU'] if config['USE_CUDA'] else 'cpu'
        embedding = [torch.Tensor(n_embeddings, embedding_dim).to(device) for _ in range(n_codebook)]
        init_bound = 1/400
        self.embedding = [embedding[i].uniform_(-init_bound, init_bound) for i in range(n_codebook)]
        self.ema_count = [torch.zeros(n_embeddings).to(device) for _ in range(n_codebook)]
        self.ema_weight = [embed.clone() for embed in self.embedding]
        self.unactivated_count = [-torch.ones(n_embeddings).to(device)  for _ in range(n_codebook)]



    def pcf_vq_embedding(self, pcf_semantic):
        # inference
        B, D = pcf_semantic.size()
        res = pcf_semantic.detach()
        embs, residuals, sem_ids = [], [], []
        for i in range(self.n_codebook):
            residuals.append(res)
            pcf_distance = torch.addmm(torch.sum(self.embedding[i] ** 2, dim=1) +
                                     torch.sum(res ** 2, dim=1, keepdim=True),
                                     res, self.embedding[i].t(),
                                     alpha=-2.0, beta=1.0)
            pcf_indices = torch.argmin(pcf_distance.double(), dim=-1)
            emb = F.embedding(pcf_indices, self.embedding[i])
            res = res - emb
            sem_ids.append(pcf_indices)
            embs.append(emb)
        embeddings = torch.stack(embs, dim=-1)  # B,D,code_num
        residuals = torch.stack(residuals, dim=-1)  # B,D,code_num
        sem_ids = torch.stack(sem_ids, dim=-1)  # B,code_num

        pcf_quantized = embeddings.sum(dim=-1)
        pcf_quantized = pcf_semantic + (pcf_quantized - pcf_semantic).detach()
        return pcf_quantized, residuals, sem_ids


    def plm_vq_embedding(self, plm_semantic):
        # inference
        B, D = plm_semantic.size()
        res = plm_semantic.detach()
        embs, residuals, sem_ids = [], [], []
        for i in range(self.n_codebook):
            residuals.append(res)
            plm_distance = torch.addmm(torch.sum(self.embedding[i] ** 2, dim=1) +
                                     torch.sum(res ** 2, dim=1, keepdim=True),
                                     res, self.embedding[i].t(),
                                     alpha=-2.0, beta=1.0)
            plm_indices = torch.argmin(plm_distance.double(), dim=-1)
            emb = F.embedding(plm_indices, self.embedding[i])
            res = res - emb
            sem_ids.append(plm_indices)
            embs.append(emb)
        embeddings = torch.stack(embs, dim=-1),  # B,D,code_num
        residuals = torch.stack(residuals, dim=-1),  # B,D,code_num
        sem_ids = torch.stack(sem_ids, dim=-1)  # B,code_num

        plm_quantized = embeddings[0].sum(dim=-1) # 注意embedding
        plm_quantized = plm_semantic + (plm_quantized - plm_semantic).detach() # 这里好奇怪，为啥要这么搞
        return plm_quantized, residuals, sem_ids


    def cmcm_alignment(self, pcf_semantic, plm_semantic, i):
        # 注意哪部分可以更新，哪部分不可以
        B, D = pcf_semantic.size()
        pcf_distances_gradient = torch.addmm(torch.sum(self.embedding[i] ** 2, dim=1) +
                                             torch.sum(pcf_semantic ** 2, dim=1, keepdim=True),
                                             pcf_semantic, self.embedding[i].t(),
                                             alpha=-2.0, beta=1.0)  # [B, M]
        plm_distances_gradient = torch.addmm(torch.sum(self.embedding[i] ** 2, dim=1) +
                                             torch.sum(plm_semantic ** 2, dim=1, keepdim=True),
                                             plm_semantic, self.embedding[i].t(),
                                             alpha=-2.0, beta=1.0)  # [B, M]
        pcf_ph = F.softmax(-torch.sqrt(pcf_distances_gradient), dim=1)  # [B, M] torch.Size([160, 512])
        plm_ph = F.softmax(-torch.sqrt(plm_distances_gradient), dim=1)  # [B, M] torch.Size([160, 512])
        pcf_pH = pcf_ph
        plm_pH = plm_ph

        Scode = pcf_pH @ torch.log(plm_pH.t() + 1e-10) + plm_pH @ torch.log(pcf_pH.t() + 1e-10)  # B,B
        MaxScode = torch.max(-Scode)
        EScode = torch.exp(Scode + MaxScode)
        EScode_sumdim1 = torch.sum(EScode, dim=1)
        Lcmcm = 0
        for i in range(B):
            Lcmcm -= torch.log(EScode[i, i] / (EScode_sumdim1[i] + self.epsilon))
        Lcmcm /= B  # 这一步是啥意思。这个用于算CMCM loss，暂时先不要
        return Lcmcm # [.,.,.]

    def perplexity(self, pcf_encodings, plm_encodings):
        # 计算code困惑度，这个放到for循环中
        pcf_avg_probs = torch.mean(pcf_encodings, dim=0)
        pcf_perplexity = torch.exp(-torch.sum(pcf_avg_probs * torch.log(pcf_avg_probs + 1e-10)))
        plm_avg_probs = torch.mean(plm_encodings, dim=0)
        plm_perplexity = torch.exp(-torch.sum(plm_avg_probs * torch.log(plm_avg_probs + 1e-10)))
        return pcf_perplexity, plm_perplexity

    def forward(self, pcf_semantic, plm_semantic):
        # training
        M, D = self.embedding[0].size() # Code_num, H
        pcf_res = pcf_semantic.detach()
        plm_res = plm_semantic.detach()

        pcf_embs, pcf_residuals, pcf_sem_ids = [], [], []
        plm_embs, plm_residuals, plm_sem_ids = [], [], []
        Lcmcms , pcf_perplexitys, plm_perplexitys= [], [], []
        for i in range(self.n_codebook): # Code level
            Lcmcm = self.cmcm_alignment(pcf_semantic, plm_semantic, i)
            Lcmcms.append(Lcmcm)

            pcf_residuals.append(pcf_res) # B,H
            plm_residuals.append(plm_res)

            pcf_distances = torch.addmm(torch.sum(self.embedding[i] ** 2, dim=1) +
                                        torch.sum(pcf_res ** 2, dim=1, keepdim=True),
                                        pcf_res, self.embedding[i].t(),
                                        alpha=-2.0, beta=1.0)  # [B, M]

            plm_distances = torch.addmm(torch.sum(self.embedding[i] ** 2, dim=1) +
                                        torch.sum(plm_res ** 2, dim=1, keepdim=True),
                                        plm_res, self.embedding[i].t(),
                                        alpha=-2.0, beta=1.0)  # [B, M]

            pcf_indices = torch.argmin(pcf_distances, dim=-1)  # [B,1], .double()
            # print("XXXXX", pcf_indices)
            pcf_encodings = F.one_hot(pcf_indices, M).float() # [B, M] .double()
            pcf_emb = F.embedding(pcf_indices, self.embedding[i])

            plm_indices = torch.argmin(plm_distances, dim=-1)  # [B,1] .double()
            plm_encodings = F.one_hot(plm_indices, M).float()  # [B, M] .double()
            plm_emb = F.embedding(plm_indices, self.embedding[i]) # [B,M]
            # plm_res = plm_res - plm_emb

            pcf_embs.append(pcf_emb)
            plm_embs.append(plm_emb)

            pcf_sem_ids.append(pcf_indices)
            plm_sem_ids.append(plm_indices)


            if self.training: # EMA更新
                # pcf

                self.ema_count[i] = self.decay * self.ema_count[i] + (1 - self.decay) * torch.sum(pcf_encodings, dim=0) # 每个code的个数, 纵向
                # self.ema_count[i] = torch.clamp(self.ema_count[i], min=1e-8)

                n = torch.sum(self.ema_count[i]) # 总code的个数
                # print("XXXXXXX", n==0)
                self.ema_count[i] = (self.ema_count[i] + self.epsilon) / (n + M * self.epsilon) * n # norm
                # print(pcf_encodings.dtype, pcf_res.dtype)
                pcf_dw = torch.matmul(pcf_encodings.t(), pcf_res) # M*D， 每种code对应embedding的和,可以换成attn

                # ********************************************************
                pcf_plm_dw = torch.matmul(pcf_encodings.t(), plm_res) # M*B * B*D = M*D，另一种模态的和
                # ********************************************************

                self.ema_weight[i] = self.decay * self.ema_weight[i] + 0.5 * (1 - self.decay) * pcf_dw + 0.5 * (
                            1 - self.decay) * pcf_plm_dw # o
                # print("Before",i,torch.isnan(self.ema_weight[i]).any())
                self.embedding[i] = self.ema_weight[i] / self.ema_count[i].unsqueeze(-1)
                # print("After",i,torch.isnan(self.embedding[i]).any())


                # plm
                self.ema_count[i] = self.decay * self.ema_count[i] + (1 - self.decay) * torch.sum(plm_encodings, dim=0)
                n = torch.sum(self.ema_count[i])
                self.ema_count[i] = (self.ema_count[i] + self.epsilon) / (n + M * self.epsilon) * n
                plm_dw = torch.matmul(plm_encodings.t(), plm_res)
                # ********************************************************
                plm_pcf_dw = torch.matmul(plm_encodings.t(), pcf_res)
                # ********************************************************
                self.ema_weight[i] = self.decay * self.ema_weight[i] + 0.5 * (1 - self.decay) * plm_dw + 0.5 * (
                            1 - self.decay) * plm_pcf_dw
                self.embedding[i] = self.ema_weight[i] / self.ema_count[i].unsqueeze(-1) # 万一没有不就是NAN?, 因为这里

            # reset popular code
            self.unactivated_count[i] += 1  # 每次都加1
            for indice in pcf_indices:
                self.unactivated_count[i][indice.item()] = 0 # 激活就置为0，有一方激活即可
            for indice in plm_indices:
                self.unactivated_count[i][indice.item()] = 0
            activated_indices = []
            unactivated_indices = []
            for j, x in enumerate(self.unactivated_count[i]):
                if x > 300: # 一直都没有被置为0，连续300次没有激活，就重新初始化
                    unactivated_indices.append(j)
                    self.unactivated_count[i][j] = 0
                elif x >= 0 and x < 100:
                    activated_indices.append(j)
            activated_quantized = F.embedding(torch.tensor(activated_indices, dtype=torch.int32).to(pcf_emb.device), self.embedding[i])
            for j in unactivated_indices: # 重新初始化
                self.embedding[i][j] = activated_quantized[random.randint(0, len(activated_indices) - 1)] + torch.Tensor(
                    self.embedding_dim).uniform_(-1 / 1024, 1 / 1024).to(pcf_semantic.device) # 256维度

            pcf_perplexity, plm_perplexity = self.perplexity(pcf_encodings, plm_encodings)
            pcf_perplexitys.append(pcf_perplexity)
            plm_perplexitys.append(plm_perplexity)

            pcf_res = pcf_res - pcf_emb # 残差，下一次
            plm_res = plm_res - plm_emb



        # 从这往下是计算损失，不用包含在for循环内

        pcf_embeddings = torch.stack(pcf_embs, dim=-1)  # B,H,code_num
        pcf_residuals = torch.stack(pcf_residuals, dim=-1)  # B,H,code_num
        pcf_sem_ids = torch.stack(pcf_sem_ids, dim=-1)  # B,code_num
        
        plm_embeddings = torch.stack(plm_embs, dim=-1)  # B,H,code_num
        plm_residuals = torch.stack(plm_residuals, dim=-1)  # B,H,code_num
        plm_sem_ids = torch.stack(plm_sem_ids, dim=-1)  # B,code_num

        pcf_quantized = pcf_embeddings.sum(dim=-1)
        plm_quantized = plm_embeddings.sum(dim=-1)

        assert pcf_quantized.size() == pcf_semantic.size() # 不知道上面那个什么鬼。
        # print("AAAAAAAAXXXXXX",torch.isnan(pcf_quantized).any())
        # print(torch.isnan(self.embedding[0]).any())
        # print(torch.isnan(self.embedding[1]).any())
        # print(torch.isnan(self.embedding[2]).any())


        cmcm_loss = 0.5 * Lcmcm.sum(dim=-1)

        # commitment loss
        pcf_e_latent_loss = F.mse_loss(pcf_semantic, pcf_quantized.detach())
        pcf_plm_e_latent_loss = F.mse_loss(pcf_semantic, plm_quantized.detach())
        pcf_loss = self.commitment_cost * 2.0 * pcf_e_latent_loss + self.commitment_cost * pcf_plm_e_latent_loss
        plm_e_latent_loss = F.mse_loss(plm_semantic, plm_quantized.detach())
        plm_pcf_e_latent_loss = F.mse_loss(plm_semantic, pcf_quantized.detach())
        plm_loss = self.commitment_cost * 2.0 * plm_e_latent_loss + self.commitment_cost * plm_pcf_e_latent_loss

        pcf_quantized = pcf_semantic + (pcf_quantized - pcf_semantic).detach() # residue B,H
        plm_quantized = plm_semantic + (plm_quantized - plm_semantic).detach()

        pcf_perplexity = sum(pcf_perplexitys) / self.n_codebook
        plm_perplexity = sum(plm_perplexitys) / self.n_codebook # metric不需要加和


        return pcf_quantized, plm_quantized, pcf_loss, plm_loss, pcf_perplexity, plm_perplexity,cmcm_loss #, equal_num,这个eual


##### Decoder


class TaskNCELayer(nn.Module):
    def __init__(self, emb_dim, hidden_dim, inner_weight=0.1, outer_weight=0.01, dropout=0.2): # 0.1, 0.01
        super(TaskNCELayer, self).__init__()
        self.inner_weight = inner_weight
        self.outer_weight = outer_weight

        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax(dim=-1)

        self.pcf_proj = nn.Linear(emb_dim, hidden_dim)
        self.plm_proj = nn.Linear(emb_dim, hidden_dim)

        self.pcf_out = nn.Linear(hidden_dim, emb_dim)
        self.plm_out = nn.Linear(hidden_dim, emb_dim)

        self.pcf_ref_proj = nn.Dropout(dropout)# 这里使用LSTM/SUM pooling/Attention
        self.plm_ref_proj = nn.Dropout(dropout)# 这里使用LSTM/SUM pooling/Attention
        self.bn = nn.BatchNorm1d(emb_dim, affine=False)
    
    def bn_dot(self,a,b):
        if a.shape[0]>1:
            return self.bn(a).T @ self.bn(b)
        else:
            return a.T @ b
    def forward_corr(self, pcf_vq, plm_vq, pcf_ref, plm_ref, temp=0.005): # 模仿进行code替换; nn.BatchNorm1d(num_feats, affine=False)
        """这里也可以升级成batch内negative， 模拟随机替换
        https://github.com/val-iisc/NoisyTwins/blob/3f686b32c9af7d86bac3182a04c382671abe33a8/src/utils/barlow.py#L81
        """
        batch_dim, emb_dim = pcf_vq.size()
        pcf_ref_pos, pcf_ref_neg = pcf_ref
        plm_ref_pos, plm_ref_neg = plm_ref
        # empirical cross-correlation matrix
        c = self.bn_dot(pcf_vq, pcf_ref_pos)# self.bn(pcf_vq).T @ self.bn(pcf_ref_pos)
        # sum the cross-correlation matrix between all gpus
        c.div_(pcf_vq.shape[0])
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        pcf_inner_loss = on_diag + temp * off_diag

        c = self.bn_dot(plm_vq, plm_ref_pos)#self.bn(plm_vq).T @ self.bn(plm_ref_pos)
        # sum the cross-correlation matrix between all gpus
        c.div_(plm_vq.shape[0])
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        plm_inner_loss = on_diag + temp * off_diag

        c = self.bn_dot(pcf_vq, plm_ref_pos)#self.bn(pcf_vq).T @ self.bn(plm_ref_pos)
        # sum the cross-correlation matrix between all gpus
        c.div_(pcf_vq.shape[0])
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        pcf_outer_loss = on_diag + temp * off_diag
        c =self.bn_dot(plm_vq, pcf_ref_pos)# self.bn(plm_vq).T @ self.bn(pcf_ref_pos)
        # sum the cross-correlation matrix between all gpus
        c.div_(plm_vq.shape[0])
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        plm_outer_loss = on_diag + temp * off_diag

        cor_loss = self.inner_weight * (pcf_inner_loss + plm_inner_loss) + self.outer_weight * (pcf_outer_loss + plm_outer_loss)
        return cor_loss

    def forward(self, pcf_vq, plm_vq, pcf_ref, plm_ref): # 这个其实本质上是静态的，想搞成动态的。ref=(positive, negative)
        """这里也可以升级成batch内negative"""
        # pcf_vq = F.normalize(pcf_vq, p=2, dim=1)
        # plm_vq = F.normalize(plm_vq, p=2, dim=1)

        loss_cor = self.forward_corr(pcf_vq, plm_vq, pcf_ref, plm_ref)
        batch_dim, emb_dim = pcf_vq.size()
        
        pcf_ref_pos, pcf_ref_neg = pcf_ref # B, H
        plm_ref_pos, plm_ref_neg = plm_ref

        # pos_pcf, _ = self.pcf_ref_proj(pcf_ref).sum(dim=1) # B,D, neg不一定能用到
        # pos_plm, _ = self.plm_ref_proj(plm_ref).sum(dim=1)
        #
        # pcf_cxt, plm_cxt = self.pcf_proj(pcf_vq), self.plm_proj(plm_vq) # B,D
        # pcf_cxt, plm_cxt = self.pcf_out(pcf_cxt), self.plm_out(plm_cxt) # B,D

        # feature based contrastive learning :soft，也可以使用明确的negative
        # inner contrastive
        inner_nce = 0
        pcf_inner = torch.matmul(pcf_vq, pcf_ref_pos.t()) # B,K , 这里的对比学习好奇怪
        plm_inner = torch.matmul(plm_vq, plm_ref_pos.t())
        inner_nce = torch.sum(torch.diag(self.lsoftmax(pcf_inner))) + torch.sum(torch.diag(self.lsoftmax(plm_inner)))

        # outer contrastive
        outer_nce = 0
        pcf_outer = torch.matmul(pcf_vq, plm_ref_pos.t()) # B,B
        plm_outer = torch.matmul(plm_vq, pcf_ref_pos.t())
        outer_nce = torch.sum(torch.diag(self.lsoftmax(pcf_outer))) + torch.sum(torch.diag(self.lsoftmax(plm_outer)))

        soft_nce = self.inner_weight * inner_nce + self.outer_weight * outer_nce
        soft_nce /= -1.*batch_dim

        # print("AAAAAAAA",inner_nce, outer_nce, torch.isnan(pcf_inner).any(),torch.isnan(plm_vq).any(), torch.isnan(pcf_vq).any(), torch.isnan(pcf_ref_pos).any())
        # hard_nce # 暂时不管
        return soft_nce + loss_cor



class CondNorm(nn.Module):
    """鼓励相同code的产生不同的表示"""
    def __init__(self, input_dim, output_dim, norm_layer=nn.GroupNorm, f_channels=128, freeze_norm_layer=False, add_linear=False,  **norm_layer_params):
        super(CondNorm, self).__init__()
        # self.norm_layer = norm_layer(input_dim, **norm_layer_params)
        self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)

        self.add_linear = add_linear
        if self.add_linear:
            self.linear_add = nn.Linear(input_dim, input_dim)
        if freeze_norm_layer:
            for p in self.norm_layer.parameters:
                p.requires_grad = False
        self.linear_y = nn.Linear(input_dim, output_dim)
        self.linear_b = nn.Linear(input_dim, output_dim)


    def forward(self, cond, zq):
        # 注意这里cond应该已经attention过了
        if self.add_linear:
            zq = self.linear_add(zq) # 相同的量化项产生不同的结果
        norm_cond = self.norm_layer(cond) # 如果cond是全0，就不行，尤其是drug
        new_cond = norm_cond * self.linear_y(zq) + self.linear_b(zq) # residue
        return new_cond


class PCF_Decoder(nn.Module):
    def __init__(self, hidden, init_dim, vq_dim):
        super(PCF_Decoder, self).__init__()
        self.input_dim = hidden
        self.output_dim = init_dim
        self.relu = nn.ReLU()
        self.pcf_rec = nn.Sequential(nn.Linear(hidden * 2, init_dim))#,nn.ReLU(),  nn.Linear(hidden * 3, init_dim))
        self.pcf_linear = nn.Linear(vq_dim, hidden)

        self.cond_norm = CondNorm(hidden, hidden, norm_layer=nn.GroupNorm, f_channels=hidden, freeze_norm_layer=False, add_linear=True, num_groups=4, eps=1e-6, affine=True)

    def forward(self, pcf_encoder_result, pcf_vq, pcf_cond):
        pcf_vq_result = self.pcf_linear(pcf_vq) # B, H
        # print("AAAABBBB", torch.isnan(pcf_vq_result).any())

        # 加入condition
        # print('AAAAAAAAA', pcf_cond.shape, pcf_vq_result.shape)
        pcf_vq_result = self.cond_norm(pcf_cond, pcf_vq_result) # B,H, B,H; -> B,H
        # print("BBBBBBBBB", torch.isnan(pcf_cond).any(), torch.isnan(pcf_vq_result).any())
        pcf_encoder_result = torch.cat([pcf_vq_result, pcf_encoder_result], dim=-1) # residule
        pcf_decoder_result = self.pcf_rec(pcf_encoder_result)

        return pcf_decoder_result


class PLM_Decoder(nn.Module):
    def __init__(self, hidden, init_dim, vq_dim):
        super(PLM_Decoder, self).__init__()
        self.input_dim = hidden
        self.output_dim = init_dim
        self.relu = nn.ReLU()
        self.plm_rec = nn.Linear(hidden * 2, init_dim)#MLP(hidden*2, hidden, init_dim) #; nn.Sequential(nn.Linear(hidden * 2, hidden * 3), nn.ReLU(), nn.Linear(hidden * 3, init_dim))
        self.plm_linear = nn.Linear(vq_dim, hidden)

        # self.cond_norm = CondNorm(hidden, hidden, norm_layer=nn.GroupNorm, freeze_norm_layer=False, add_linear=True)
        self.cond_norm = CondNorm(hidden, hidden, norm_layer=nn.GroupNorm, f_channels=hidden, freeze_norm_layer=False, add_linear=True, num_groups=4, eps=1e-6, affine=True)

    def forward(self, plm_encoder_result, plm_vq, plm_cond):
        plm_vq_result = self.plm_linear(plm_vq)

        # 加入condition
        plm_vq_result = self.cond_norm(plm_cond, plm_vq_result) # B,D

        plm_encoder_result = torch.cat([plm_vq_result, plm_encoder_result], dim=-1)
        plm_decoder_result = self.plm_rec(plm_encoder_result)
        return plm_decoder_result

    

###### Pretrain DRL
class DRL_Encoder(nn.Module):
    """align collaborative-content"""
    def __init__(self, pcf_dim, plm_dim, pcf_output_dim, plm_output_dim, n_codebook, n_embeddings, embedding_dim, config):
        super(DRL_Encoder, self).__init__()
        self.pcf_dim = pcf_dim #  H
        self.plm_dim = plm_dim
        self.pcf_output_dim = pcf_output_dim
        self.plm_output_dim = plm_output_dim

        self.n_codebook = n_codebook
        self.n_embeddings = n_embeddings # codebook
        self.hidden_dim = embedding_dim

        self.pcf_encoder = PCF_Encoder(self.pcf_dim, self.pcf_output_dim)
        self.plm_encoder = PLM_Encoder(self.plm_dim, self.plm_output_dim)
        self.cross_quant = CrossQuant(self.n_embeddings, self.n_codebook, self.hidden_dim, config=config)

    def pcf_vq_encoder(self, pcf_feature):
        # inference, emb
        pcf_encoder_result = self.pcf_encoder(pcf_feature)
        pcf_vq = self.cross_quant.pcf_vq_embedding(pcf_encoder_result)
        return pcf_vq[0]

    def plm_vq_encoder(self, plm_feature):
        plm_encoder_result = self.plm_encoder(plm_feature)
        plm_vq = self.cross_quant.plm_vq_embedding(plm_encoder_result)
        return plm_vq[0]

    def pcf_vq_forward(self, pcf_feature, plm_feature):
        # loss
        plm_vq = self.plm_vq_encoder(plm_feature)
        pcf_encoder_result = self.pcf_encoder(pcf_feature)
        pcf_vq = self.cross_quant.pcf_vq_embedding(pcf_encoder_result)
        pcf_vq_forward_loss = F.mse_loss(pcf_encoder_result, pcf_vq.detach()) + 0.25*F.mse_loss(pcf_encoder_result, plm_vq.detach())
        return pcf_vq_forward_loss

    def plm_vq_forward(self, pcf_feature, plm_feature):
        pcf_vq = self.pcf_vq_encoder(pcf_feature)
        plm_encoder_result = self.plm_encoder(plm_feature)
        plm_vq = self.cross_quant.plm_vq_embedding(plm_encoder_result)
        plm_vq_forward_loss = F.mse_loss(plm_encoder_result, plm_vq.detach()) + 0.25*F.mse_loss(plm_encoder_result, pcf_vq.detach())
        return plm_vq_forward_loss

    def forward(self, pcf_feature, plm_feature):
        # semantic会做额外分类任务，暂时不用。encoder是specific，而semantic才是general (这里只需要share)
        pcf_encoder_result = self.pcf_encoder(pcf_feature) # B, Hidden
        plm_encoder_result = self.plm_encoder(plm_feature) # B, Hidden
        pcf_vq, plm_vq, pcf_embedding_loss, plm_embedding_loss, pcf_perplexity, plm_perplexity, cmcm_loss\
            = self.cross_quant(pcf_encoder_result, plm_encoder_result) # vq就是索引到的code

        return pcf_encoder_result, plm_encoder_result, pcf_vq, plm_vq, pcf_embedding_loss, plm_embedding_loss, \
                pcf_perplexity, plm_perplexity, cmcm_loss

class DRL_Decoder(nn.Module):
    def __init__(self, pcf_dim, plm_dim, pcf_output_dim, plm_output_dim, embedding_dim):
        super(DRL_Decoder, self).__init__()
        self.pcf_dim = pcf_dim # D
        self.plm_dim = plm_dim # 768
        self.pcf_output_dim = pcf_output_dim # H
        self.plm_output_dim = plm_output_dim # H
        self.embedding_dim = embedding_dim # H
        self.pcf_decoder = PCF_Decoder(self.pcf_output_dim, self.pcf_dim, self.embedding_dim)
        self.plm_decoder = PLM_Decoder(self.plm_output_dim, self.plm_dim, self.embedding_dim)


    def forward(self, pcf_feat, plm_feat, pcf_encoder_result, plm_encoder_result, pcf_vq, plm_vq, pcf_cond, plm_cond):
        pcf_recon_result = self.pcf_decoder(pcf_encoder_result, pcf_vq, pcf_cond)
        plm_recon_result = self.plm_decoder(plm_encoder_result, plm_vq, plm_cond)
        pcf_recon_loss = F.mse_loss(pcf_recon_result, pcf_feat)
        plm_recon_loss = F.mse_loss(plm_recon_result, plm_feat) # 放大这一侧的损失
        return pcf_recon_loss, plm_recon_loss, pcf_recon_result, plm_recon_result


class DRL(nn.Module):
    def __init__(self, pcf_embedding, plm_embedding, pcf_proc_cond_embedding, plm_proc_cond_embedding, pcf_drug_cond_embedding, plm_drug_cond_embedding, config, mode=None):
        super(DRL, self).__init__()
        if config['USE_CUDA']==True:
            self.device = 'cuda:' + config['GPU']
        else:
            self.device ='cpu'

        self.mode = mode
        self.task = config['TASK']
        self.pcf_embedding_w = pcf_embedding # DIM
        self.plm_embedding_w = plm_embedding # 768
        self.pcf_proc_cond_embedding_w = pcf_proc_cond_embedding # 如果没训练好，很多pretrain的就是0
        self.plm_proc_cond_embedding_w = plm_proc_cond_embedding
        self.pcf_drug_cond_embedding_w = pcf_drug_cond_embedding
        self.plm_drug_cond_embedding_w = plm_drug_cond_embedding

        # config
        self.pcf_dim = config['DIM']
        self.plm_dim = 768
        self.pcf_output_dim = config['HIDDEN']
        self.plm_output_dim = config['HIDDEN']
        self.embedding_dim = config['HIDDEN'] # codebook
        self.n_codebook = config['N_CODEBOOK']
        self.n_embeddings = config['N_EMBED'] # code num

        self.init_embedding()

        self.proj_pcf = nn.Linear(self.pcf_dim, config['HIDDEN'])
        self.proj_plm = nn.Linear(768, config['HIDDEN']) # 固定维度
        self.pcf_cond_attention = nn.MultiheadAttention(embed_dim=self.pcf_output_dim, num_heads=2, batch_first=True)
        self.plm_cond_attention = nn.MultiheadAttention(embed_dim=self.plm_output_dim, num_heads=2, batch_first=True)
        self.pcf_ref_attention = nn.MultiheadAttention(embed_dim=self.pcf_output_dim, num_heads=2, batch_first=True)
        self.plm_ref_attention = nn.MultiheadAttention(embed_dim=self.plm_output_dim, num_heads=2, batch_first=True)

        self.encoder = DRL_Encoder(self.pcf_dim, self.plm_dim, self.pcf_output_dim, self.plm_output_dim, self.n_codebook, self.n_embeddings, self.embedding_dim, config)
        self.decoder = DRL_Decoder(self.pcf_dim, self.plm_dim, self.pcf_output_dim, self.plm_output_dim, self.embedding_dim)
        self.task_aware = TaskNCELayer(self.embedding_dim, self.embedding_dim)

    def cond_emb(self, pcf_proc_cond_emb, plm_proc_cond_emb, pcf_drug_cond_emb, plm_drug_cond_emb, proc_mask, drug_mask):
        """Transfer into one vector"""
        proc_mask_a = proc_mask#.float().masked_fill(proc_mask == 0, -1e9).masked_fill(proc_mask == 1, float(0.0)) # 不大一样
        drug_mask_a = drug_mask#.float().masked_fill(drug_mask == 0, -1e9).masked_fill(drug_mask == 1, float(0.0)) # 不大一样，必须得有，不然会出现NAN在Attention中

        # 找到 mask 全为 False 的行
        all_false_rows_p = torch.all(proc_mask_a == False, dim=1)
        all_false_rows_d = torch.all(drug_mask_a == False, dim=1)

        # 将这些行的 mask 替换为全 True
        proc_mask_a[all_false_rows_p] = True
        drug_mask_a[all_false_rows_d] = True


        # print(proc_mask_a[0])
        # print(pcf_proc_cond_emb[0])

        # emb transfer
        pcf_proc_cond_emb = self.proj_pcf(pcf_proc_cond_emb) # B*V, T, D
        plm_proc_cond_emb = self.proj_plm(plm_proc_cond_emb)
        pcf_drug_cond_emb = self.proj_pcf(pcf_drug_cond_emb)
        plm_drug_cond_emb = self.proj_plm(plm_drug_cond_emb)

        pcf_proc_cond_emb,_ = self.pcf_cond_attention(pcf_proc_cond_emb, pcf_proc_cond_emb, pcf_proc_cond_emb, key_padding_mask=~proc_mask_a) # B,T,D
        plm_proc_cond_emb,_ = self.plm_cond_attention(plm_proc_cond_emb, plm_proc_cond_emb, plm_proc_cond_emb, key_padding_mask=~proc_mask_a) # B,T,D
        pcf_proc_cond_emb[all_false_rows_p] = 0
        plm_proc_cond_emb[all_false_rows_p] = 0

        # pcf_proc_cond = F.avg_pool1d(pcf_proc_cond_emb.permute(0,2,1), kernel_size=pcf_proc_cond_emb.size(1)).squeeze(2) # B,D
        # plm_proc_cond = F.avg_pool1d(plm_proc_cond_emb.permute(0,2,1), kernel_size=plm_proc_cond_emb.size(1)).squeeze(2) # B,D

        # print("A", pcf_proc_cond_emb.shape, proc_mask.shape, drug_mask)
        pcf_proc_cond = get_last_visit(pcf_proc_cond_emb, mask=proc_mask) # proc_mask, 如果是全0怎么办，因为真实情况序列对应的就是全0
        plm_proc_cond = get_last_visit(plm_proc_cond_emb, mask=proc_mask) # proc_mask

        # drug在Rec任务中可能出现mask全0，所以这里必须要有mask?
        # print("CCCCCCC", drug_mask.shape)
        # if torch.any(drug_mask.sum(dim=1) == 0):
        #     print("Zero found, breaking out.")


        pcf_drug_cond_emb,_ = self.pcf_cond_attention(pcf_drug_cond_emb, pcf_drug_cond_emb, pcf_drug_cond_emb, key_padding_mask=~drug_mask_a) # B,T,D
        plm_drug_cond_emb,_ = self.plm_cond_attention(plm_drug_cond_emb, plm_drug_cond_emb, plm_drug_cond_emb, key_padding_mask=~drug_mask_a) # B,T,D
        # pcf_drug_cond = F.avg_pool1d(pcf_drug_cond_emb.permute(0,2,1), kernel_size=pcf_drug_cond_emb.size(1)).squeeze(2) # B,D
        # plm_drug_cond = F.avg_pool1d(plm_drug_cond_emb.permute(0,2,1), kernel_size=plm_drug_cond_emb.size(1)).squeeze(2) # B,D
        # print(pcf_drug_cond_emb, drug_mask)
        pcf_drug_cond_emb[all_false_rows_d] = 0
        plm_drug_cond_emb[all_false_rows_d] = 0

        pcf_drug_cond = get_last_visit(pcf_drug_cond_emb, mask=drug_mask) # drug_mask
        plm_drug_cond = get_last_visit(plm_drug_cond_emb, mask=drug_mask) # drug_mask, 这里有些; layernorm出现nan

        pcf_cond = pcf_proc_cond #+ pcf_drug_cond # B,D 可能会出现加和为inf会出现无穷大。
        plm_cond = plm_proc_cond #+ plm_drug_cond

        return pcf_cond, plm_cond

    def ref_emb(self, pcf_pos_ref, pcf_neg_ref, plm_pos_ref, plm_neg_ref, pos_diag_mask, neg_diag_mask):
        # 也可以先加一个头部cls
        pcf_pos_ref = self.proj_pcf(pcf_pos_ref)
        pcf_neg_ref = self.proj_pcf(pcf_neg_ref)
        plm_pos_ref = self.proj_plm(plm_pos_ref)
        plm_neg_ref = self.proj_plm(plm_neg_ref)


        # 找到 mask 全为 False 的行
        all_false_rows_p = torch.all(pos_diag_mask == False, dim=1)
        all_false_rows_n = torch.all(neg_diag_mask == False, dim=1)

        # 将这些行的 mask 替换为全 True
        pos_diag_mask[all_false_rows_p] = True
        neg_diag_mask[all_false_rows_n] = True

        pcf_pos_ref,_ = self.pcf_ref_attention(pcf_pos_ref, pcf_pos_ref, pcf_pos_ref, key_padding_mask=~pos_diag_mask) # B,T,D
        plm_pos_ref,_ = self.plm_ref_attention(plm_pos_ref, plm_pos_ref, plm_pos_ref, key_padding_mask=~pos_diag_mask) # B,T,D

        pcf_pos_ref[all_false_rows_p] = 0
        plm_pos_ref[all_false_rows_p] = 0

        # print("AAAAAAA", pos_diag_mask.shape, pcf_pos_ref.shape)
        pcf_pos_ref = pos_diag_mask.unsqueeze(-1) * pcf_pos_ref
        plm_pos_ref = pos_diag_mask.unsqueeze(-1) * plm_pos_ref
        pcf_pos_ref = pcf_pos_ref.sum(dim=1)
        plm_pos_ref = plm_pos_ref.sum(dim=1)
        # pcf_pos_ref = get_last_visit(pcf_pos_ref, pos_diag_mask)
        # plm_pos_ref = get_last_visit(plm_pos_ref, pos_diag_mask)


        pcf_neg_ref,_ = self.pcf_ref_attention(pcf_neg_ref, pcf_neg_ref, pcf_neg_ref, key_padding_mask=~neg_diag_mask) # B,T,D
        plm_neg_ref,_ = self.plm_ref_attention(plm_neg_ref, plm_neg_ref, plm_neg_ref, key_padding_mask=~neg_diag_mask) # B,T,D

        pcf_neg_ref[all_false_rows_n] = 0
        plm_neg_ref[all_false_rows_n] = 0

        pcf_neg_ref = get_last_visit(pcf_neg_ref, neg_diag_mask)
        plm_neg_ref = get_last_visit(plm_neg_ref, neg_diag_mask)
        return pcf_pos_ref, pcf_neg_ref, plm_pos_ref, plm_neg_ref


    def init_embedding(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0)
            self.apply(_init_weights)

        self.pcf_embedding = nn.Embedding(self.pcf_embedding_w.shape[0], self.pcf_embedding_w.shape[1])
        self.plm_embedding = nn.Embedding(self.plm_embedding_w.shape[0], self.plm_embedding_w.shape[1])
        self.pcf_proc_cond_embedding = nn.Embedding(self.pcf_proc_cond_embedding_w.shape[0], self.pcf_proc_cond_embedding_w.shape[1], padding_idx=0) # 同样0值不更新
        self.plm_proc_cond_embedding = nn.Embedding(self.plm_proc_cond_embedding_w.shape[0], self.plm_proc_cond_embedding_w.shape[1], padding_idx=0)
        self.pcf_drug_cond_embedding = nn.Embedding(self.pcf_drug_cond_embedding_w.shape[0], self.pcf_drug_cond_embedding_w.shape[1], padding_idx=0)
        self.plm_drug_cond_embedding = nn.Embedding(self.plm_drug_cond_embedding_w.shape[0], self.plm_drug_cond_embedding_w.shape[1], padding_idx=0)
        
        self.pcf_embedding.weight.data.copy_(self.pcf_embedding_w)
        self.plm_embedding.weight.data.copy_(self.plm_embedding_w)
        self.pcf_proc_cond_embedding.weight.data.copy_(self.pcf_proc_cond_embedding_w)
        self.plm_proc_cond_embedding.weight.data.copy_(self.plm_proc_cond_embedding_w)
        self.pcf_drug_cond_embedding.weight.data.copy_(self.pcf_drug_cond_embedding_w)
        self.plm_drug_cond_embedding.weight.data.copy_(self.plm_drug_cond_embedding_w)
        self.pcf_embedding.weight.requires_grad = False
        self.plm_embedding.weight.requires_grad = False
        self.pcf_proc_cond_embedding.weight.requires_grad = False # 这几个是不动的。 joint training是摇动的
        self.pcf_drug_cond_embedding.weight.requires_grad = False
        self.plm_proc_cond_embedding.weight.requires_grad = False
        self.plm_drug_cond_embedding.weight.requires_grad = False

    def find_nearest_neighbors_and_sum(self, A, B, k_neighbors=5):
        """
        在给定的 D*K 张量中找到与 B*K 张量最相近的 K 个邻居，并计算其表征的和。

        参数:
        A (torch.Tensor): D*K 张量
        B (torch.Tensor): B*K 张量
        k_neighbors (int): 需要找到的邻居数量

        返回:
        torch.Tensor: 最近邻居的表征和
        """
        # 计算欧几里得距离
        distances = torch.cdist(B, A)  # 计算 B 中每个样本到 A 中每个样本的距离，形状为 (B, D)

        # 找到最近的 k_neighbors 个邻居的索引
        _, nearest_indices = torch.topk(distances, k_neighbors, largest=False)

        # 计算邻居的表征和
        neighbors_sum = A[nearest_indices].sum(dim=1)  # 形状为 (B, K)

        return neighbors_sum

    def predict(self, diag_code, proc_code, drug_code, diag_emb=None, proc_emb=None, drug_emb=None):
        """或许学习到的是个分布，要学会去编码，而不是直接替换。"""
        # 这里和forward有啥不一样
        # B,V,M
        b, v, m = diag_code.size()

        # pcf_feature = self.pcf_embedding(diag_code) # 这里出问题，后面诊断一下
        pcf_feature = diag_emb
        plm_feature = self.plm_embedding(diag_code)

        pcf_encoder_result = self.encoder.pcf_encoder(pcf_feature)
        plm_feature = plm_feature.view(b*v*m, -1)
        plm_encoder_result = self.encoder.plm_encoder(plm_feature) # B,V,M,H

        pcf_vq = self.encoder.pcf_vq_encoder(pcf_feature.view(-1, self.pcf_dim)) # B*V*M, H
        plm_vq = self.encoder.plm_vq_encoder(plm_feature.view(-1, self.plm_dim))

        # print("AAAA", plm_feature.shape, plm_encoder_result.shape)

        proc_code = proc_code.view(b * v, -1)
        drug_code = drug_code.view(b * v, -1)
        proc_mask, drug_mask = proc_code != 0, drug_code != 0  # B,M

        # pcf_proc_cond_emb = self.pcf_proc_cond_embedding(proc_code) # B*V,M,H
        # print("AAAAAAAAA", pcf_proc_cond_emb.shape, proc_emb.shape)
        pcf_proc_cond_emb = proc_emb.view(b*v, proc_emb.shape[-2], proc_emb.shape[-1])
        plm_proc_cond_emb = self.plm_proc_cond_embedding(proc_code)
        pcf_drug_cond_emb = drug_emb.view(b*v, drug_emb.shape[-2], drug_emb.shape[-1])
        # pcf_drug_cond_emb = self.pcf_drug_cond_embedding(drug_code)
        plm_drug_cond_emb = self.plm_drug_cond_embedding(drug_code)

        # _,_,h = pcf_proc_cond_emb.size()
        pcf_cond, plm_cond = self.cond_emb(pcf_proc_cond_emb, plm_proc_cond_emb, pcf_drug_cond_emb, plm_drug_cond_emb, proc_mask, drug_mask)
        h = pcf_cond.shape[-1]
        pcf_cond, plm_cond = pcf_cond.unsqueeze(dim=1).repeat(1,m,1), plm_cond.unsqueeze(dim=1).repeat(1,m,1)
        pcf_cond, plm_cond = pcf_cond.view(-1, h), plm_cond.view(-1, h)


        pcf_encoder_result, plm_encoder_result = pcf_encoder_result.view(-1, h), plm_encoder_result.view(-1, h) # 重复

        pcf_recon = self.decoder.pcf_decoder(pcf_encoder_result, pcf_vq, pcf_cond) # for common disease
        plm_recon = self.decoder.pcf_decoder(plm_encoder_result, plm_vq, plm_cond) # for rare disease

        # print("KKKKK", torch.isnan(pcf_cond).any(), torch.isnan(pcf_proc_cond_emb).any(),torch.isnan(pcf_recon).any())


        pcf_recon = pcf_recon.view(b, v, m, -1)
        plm_recon = plm_recon.view(b, v, m, -1)

        return pcf_recon, plm_recon


    def batch_encode(self, diag_code, cond_proc, cond_drug, pos_diag, neg_diag):
        diag_code = torch.tensor(diag_code).to(self.device)
        cond_proc = pad_sequence(cond_proc, batch_first=True, padding_value=0).to(self.device)
        cond_drug = pad_sequence(cond_drug, batch_first=True, padding_value=0).to(self.device)
        pos_diag = pad_sequence(pos_diag, batch_first=True, padding_value=0).to(self.device)
        neg_diag = pad_sequence(neg_diag, batch_first=True, padding_value=0).to(self.device)
        return diag_code, cond_proc, cond_drug, pos_diag, neg_diag

    def ana(self, diag_code, cond_proc, cond_drug):
        pcf_feature = self.pcf_embedding(diag_code)  # B, D
        plm_feature = self.plm_embedding(diag_code)  # B, 768
        pcf_vq = self.encoder.pcf_vq_encoder(pcf_feature)
        plm_vq = self.encoder.plm_vq_encoder(plm_feature)
        pcf_encoder_result = self.encoder.pcf_encoder(pcf_feature)
        plm_encoder_result = self.encoder.plm_encoder(plm_feature)

        proc_mask, drug_mask = cond_proc != 0, cond_drug != 0

        pcf_proc_cond = self.pcf_proc_cond_embedding(cond_proc)
        plm_proc_cond = self.plm_proc_cond_embedding(cond_proc)
        pcf_drug_cond = self.pcf_drug_cond_embedding(cond_drug)
        plm_drug_cond = self.plm_drug_cond_embedding(cond_drug)
        # 0.78 -> 0.785 ()
        pcf_cond, plm_cond = self.cond_emb(pcf_proc_cond, plm_proc_cond, pcf_drug_cond, plm_drug_cond, proc_mask,
                                           drug_mask)  # # B, H
        pcf_recon = self.decoder.pcf_decoder(pcf_encoder_result, pcf_vq, pcf_cond) # for common disease
        plm_recon = self.decoder.pcf_decoder(plm_encoder_result, plm_vq, plm_cond) # for rare disease

        return pcf_recon, plm_recon


    def forward(self, diag_code, cond_proc, cond_drug, pos_diag, neg_diag):
        """
        :param diag_code: B,
        :param cond_proc: B,
        :param cond_drug: B,T
        :param pos_diag:
        :param neg_diag:
        :return:
        """
        diag_code, cond_proc, cond_drug, pos_diag, neg_diag = self.batch_encode(diag_code, cond_proc, cond_drug, pos_diag, neg_diag)

        pcf_feature = self.pcf_embedding(diag_code) # B, D
        plm_feature = self.plm_embedding(diag_code) # B, 768

        pos_diag_mask, neg_diag_mask, proc_mask, drug_mask = pos_diag!=0, neg_diag!=0, cond_proc!=0, cond_drug!=0

        if self.task == 'DIAG':
            pcf_pos_ref = self.pcf_embedding(pos_diag) # B T D
            plm_pos_ref = self.plm_embedding(pos_diag) # B T 768
            pcf_neg_ref = self.pcf_embedding(neg_diag)
            plm_neg_ref = self.plm_embedding(neg_diag)
        elif self.task == 'REC':
            pcf_pos_ref = self.pcf_drug_cond_embedding(pos_diag) # next drug
            plm_pos_ref = self.plm_drug_cond_embedding(pos_diag) # B T 768
            pcf_neg_ref = self.pcf_drug_cond_embedding(neg_diag)
            plm_neg_ref = self.plm_drug_cond_embedding(neg_diag)

        pcf_proc_cond = self.pcf_proc_cond_embedding(cond_proc)
        plm_proc_cond = self.plm_proc_cond_embedding(cond_proc)
        pcf_drug_cond = self.pcf_drug_cond_embedding(cond_drug)
        plm_drug_cond = self.plm_drug_cond_embedding(cond_drug)
#0.78 -> 0.785 ()
        pcf_cond, plm_cond = self.cond_emb(pcf_proc_cond, plm_proc_cond, pcf_drug_cond, plm_drug_cond, proc_mask, drug_mask) # # B, H
        pcf_pos_ref, pcf_neg_ref, plm_pos_ref, plm_neg_ref = self.ref_emb(pcf_pos_ref, pcf_neg_ref, plm_pos_ref, plm_neg_ref, pos_diag_mask, neg_diag_mask) # B,H

        pcf_encoder_result, plm_encoder_result,pcf_vq, plm_vq, pcf_embedding_loss, \
         plm_embedding_loss, pcf_perplexity, plm_perplexity, cmcm_loss = self.encoder(pcf_feature, plm_feature)

        # print("AAAAAAAAAAAAAAAA", torch.isnan(pcf_cond).any(), torch.isnan(pcf_pos_ref).any(),torch.isnan(pcf_vq).any())

        # diagnoise prediction
        task_aware_loss = self.task_aware(pcf_vq, plm_vq, [pcf_pos_ref, pcf_neg_ref], [plm_pos_ref, plm_neg_ref])


        pcf_recon_loss, plm_recon_loss, pcf_recon_result, plm_recon_result = self.decoder(pcf_feature, plm_feature, pcf_encoder_result, plm_encoder_result, pcf_vq, plm_vq, pcf_cond, plm_cond)
        loss = pcf_embedding_loss + plm_embedding_loss + pcf_recon_loss + plm_recon_loss  + cmcm_loss + task_aware_loss

        loss_item = {
            'pcf_embedding_loss': pcf_embedding_loss, # commitment loss
            'plm_embedding_loss': plm_embedding_loss,
            'pcf_recon_loss': pcf_recon_loss,
            'plm_recon_loss': plm_recon_loss,
            'task_aware_loss': task_aware_loss,
            'cmcm_loss': cmcm_loss,
            'loss': loss,
            'y_pcf_true': pcf_feature,
            'y_plm_true': plm_feature,
            'y_pcf_prob': pcf_recon_result,
            'y_plm_prob': plm_recon_result,
        }

        return loss_item



    def forward_joint(self, diag_code, cond_proc, cond_drug, pos_diag, neg_diag, pcf_embeddings):
        """ 传入pcf model的embedding，每次都换
        :param diag_code: B,
        :param cond_proc: B,
        :param cond_drug: B,T
        :param pos_diag:
        :param neg_diag:
        :return:
        """
        # 如果pcf embeddings是变化的
        self.pcf_embedding = pcf_embeddings['conditions']
        self.pcf_proc_cond_embedding = pcf_embeddings['procedures']
        self.pcf_drug_cond_embedding = pcf_embeddings['drugs'] # 用同一组embedding
        
        diag_code, cond_proc, cond_drug, pos_diag, neg_diag = self.batch_encode(diag_code, cond_proc, cond_drug, pos_diag, neg_diag)

        pcf_feature = self.pcf_embedding(diag_code) # B, D
        plm_feature = self.plm_embedding(diag_code) # B, 768

        pos_diag_mask, neg_diag_mask, proc_mask, drug_mask = pos_diag!=0, neg_diag!=0, cond_proc!=0, cond_drug!=0

        if self.task == 'DIAG':
            pcf_pos_ref = self.pcf_embedding(pos_diag) # B T D
            plm_pos_ref = self.plm_embedding(pos_diag) # B T 768
            pcf_neg_ref = self.pcf_embedding(neg_diag)
            plm_neg_ref = self.plm_embedding(neg_diag)
        elif self.task == 'REC':
            pcf_pos_ref = self.pcf_drug_cond_embedding(pos_diag) # next drug
            plm_pos_ref = self.plm_drug_cond_embedding(pos_diag) # B T 768
            pcf_neg_ref = self.pcf_drug_cond_embedding(neg_diag)
            plm_neg_ref = self.plm_drug_cond_embedding(neg_diag)

        pcf_proc_cond = self.pcf_proc_cond_embedding(cond_proc)
        plm_proc_cond = self.plm_proc_cond_embedding(cond_proc)
        pcf_drug_cond = self.pcf_drug_cond_embedding(cond_drug)
        plm_drug_cond = self.plm_drug_cond_embedding(cond_drug)
#0.78 -> 0.785 ()
        pcf_cond, plm_cond = self.cond_emb(pcf_proc_cond, plm_proc_cond, pcf_drug_cond, plm_drug_cond, proc_mask, drug_mask) # # B, H
        pcf_pos_ref, pcf_neg_ref, plm_pos_ref, plm_neg_ref = self.ref_emb(pcf_pos_ref, pcf_neg_ref, plm_pos_ref, plm_neg_ref, pos_diag_mask, neg_diag_mask) # B,H

        pcf_encoder_result, plm_encoder_result,pcf_vq, plm_vq, pcf_embedding_loss, \
         plm_embedding_loss, pcf_perplexity, plm_perplexity, cmcm_loss = self.encoder(pcf_feature, plm_feature)

        # print("AAAAAAAAAAAAAAAA", torch.isnan(pcf_cond).any(), torch.isnan(pcf_pos_ref).any(),torch.isnan(pcf_vq).any())

        # diagnoise prediction
        task_aware_loss = self.task_aware(pcf_vq, plm_vq, [pcf_pos_ref, pcf_neg_ref], [plm_pos_ref, plm_neg_ref])


        pcf_recon_loss, plm_recon_loss, pcf_recon_result, plm_recon_result = self.decoder(pcf_feature, plm_feature, pcf_encoder_result, plm_encoder_result, pcf_vq, plm_vq, pcf_cond, plm_cond)
        loss = pcf_embedding_loss + plm_embedding_loss + pcf_recon_loss + plm_recon_loss + task_aware_loss + cmcm_loss

        loss_item = {
            'pcf_embedding_loss': pcf_embedding_loss, # commitment loss
            'plm_embedding_loss': plm_embedding_loss,
            'pcf_recon_loss': pcf_recon_loss,
            'plm_recon_loss': plm_recon_loss,
            'task_aware_loss': task_aware_loss,
            'cmcm_loss': cmcm_loss,
            'loss': loss,
            'y_pcf_true': pcf_feature,
            'y_plm_true': plm_feature,
            'y_pcf_prob': pcf_recon_result,
            'y_plm_prob': plm_recon_result,
        }

        return loss_item




###### inference Learning
class UDCHealth(BaseModel): # 继承好像不太好，每次都要换新的逻辑
    """Code enhance, 需要load所有参数, 不知道要不要tuning"""
    def __init__(self, dataset, pcf_model, plm_model, drl_model,
                 feature_keys=["conditions", "procedures", "drugs"],
                 label_key="labels",
                 mode="multilabel",
                 # 下面这都是joint learning的参数
                 train_dataset=None,
                 config=None,
                 **kwargs
                 ):
        super(UDCHealth, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.pcf_model = pcf_model
        self.plm_model = plm_model
        self.drl_model = drl_model
        self.feature_keys = feature_keys
        self.label_key = label_key
        self.feat_tokenizers = self.get_feature_tokenizers() # tokenizer
        self.label_tokenizer = self.get_label_tokenizer() # 注意这里的drug可没有spec_token; 这里label索引需要加2对于正则化
        
        self.mode = mode
        self.drop = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)


        if config['JOINT']: # load drl_model的时候参数要设置下, 而且要更换对应的forward,不能load
            self.train_dataset = train_dataset
            self.ref_matrix, self.cond_proc_matrix, self.cond_drug_matrix, self.diag_name, self.proc_name, self.drug_name = self.get_matrix(config)
            self.ref_k = config['REFK']  # ground truth: negative
            self.cond_k = config['CONDK']  # condition-k
            self.linear_fea = nn.Linear(self.pcf_model.embedding_dim, self.pcf_model.embedding_dim)
            self.linear_label = nn.Sequential(nn.Linear(self.pcf_model.label_size, self.pcf_model.label_size),
                                              nn.Sigmoid())
    def get_matrix(self, config):
        # 仿照DRL
        tokenizer = get_tokenizers(self.dataset, special_tokens=['<unk>', '<pad>']) # 保持相同的tokenizer
        diag_voc_all = tokenizer['conditions'].vocabulary.token2idx # {'480.1': 2}; 不对啊，他不能使用diseaseDE ,因为label里的他看不见
        proc_voc = tokenizer['procedures'].vocabulary.token2idx # {'480.1': 2}
        drug_voc = tokenizer['drugs'].vocabulary.token2idx # {'480.1': 2}

        diag_id2_name, proc_id2_name, drug_id2_name = get_name_map(config) # {'408.1':'content'}
        if config['DATASET']=='eICU' or 'OMOP' or 'PIC': # drug不是标准的编码， OMOP感觉是concept
            drug_id2_name = tokenizer['drugs'].vocabulary.idx2token
        if config['DATASET'] == 'eICU':
            proc_id2_name = tokenizer['procedures'].vocabulary.idx2token # 他的procedure也很烦人
        all_diag_name = np.array([diag_id2_name.get(code, 'blank') for code in diag_voc_all.keys()])
        all_proc_name = np.array([proc_id2_name.get(code, 'blank') for code in proc_voc.keys()])
        all_drug_name = np.array([drug_id2_name.get(code, 'blank') for code in drug_voc.keys()])

        diag2diag, diag2proc, diag2drug, diag2nexdrug = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int)) ,defaultdict(lambda: defaultdict(int)) # 给定数量，sampling {x: (code, num)}
        last_visits = get_last_visit_sample(self.train_dataset).values() # {'patient_id': {'conditions': [], 'procedures': [], 'drugs_hist': []}
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

        ref_matrix[:2, :] = 0 # <unk>, <pad> = 0
        ref_matrix[:, :2] = 0 # # <unk>, <pad> = 0
        np.fill_diagonal(ref_matrix, 1) # 对角线为1
        cond_proc_matrix[:2, :] = 0 # <unk>, <pad> = 0
        cond_proc_matrix[:, :2] = 0 # # <unk>, <pad> = 0
        if config['TASK'] != 'REC':
            cond_drug_matrix[:2, :] = 0 # <unk>, <pad> = 0
            cond_drug_matrix[:, :2] = 0 # # <unk>, <pad> = 0； # drugrec不能直接进行，因为它前面被padding掉。或者从visit 1开始

        return ref_matrix, cond_proc_matrix, cond_drug_matrix, all_diag_name, all_proc_name, all_drug_name

    def get_drl_input_for_analysis(self, max_code_num):
        cf_batch = torch.arange(max_code_num)
        # 处理ref_matrix
        rows = self.ref_matrix[cf_batch] if len(cf_batch) > 1 else np.expand_dims(self.ref_matrix[cf_batch],
                                                                                  axis=0)  # 获取batch中所有code对应的行
        positive_mask = rows > 0
        negative_mask = rows == 0

        # 使用numpy的argwhere和random.choice进行矢量化采样
        positive_indices = np.argwhere(positive_mask)
        positive_counts = positive_mask.sum(axis=1)

        positive_sampled = [torch.from_numpy(
            np.random.choice(positive_indices[positive_indices[:, 0] == i][:, 1], self.ref_k, replace=False))
                            if positive_counts[i] > self.ref_k else torch.from_numpy(
            positive_indices[positive_indices[:, 0] == i][:, 1])
                            for i in range(len(cf_batch))]  # 如果没有则选择index为0的。

        negative_indices = np.argwhere(negative_mask)
        negative_counts = negative_mask.sum(axis=1)
        negative_sampled = [torch.from_numpy(
            np.random.choice(negative_indices[negative_indices[:, 0] == i][:, 1], self.ref_k, replace=False))
                            if negative_counts[i] > self.ref_k else torch.from_numpy(
            negative_indices[negative_indices[:, 0] == i][:, 1])
                            for i in range(len(cf_batch))]

        # 处理cond_proc_matrix和cond_drug_matrix，类似于ref_matrix
        cond_proc_batch = self.cond_proc_matrix[cf_batch] if len(cf_batch) > 1 else np.expand_dims(
            self.cond_proc_matrix[cf_batch], axis=0)
        cond_proc_mask = cond_proc_batch > 0
        cond_proc_indices = np.argwhere(cond_proc_mask)
        cond_proc_counts = cond_proc_mask.sum(axis=1)
        cond_proc_sampled = [torch.from_numpy(
            np.random.choice(cond_proc_indices[cond_proc_indices[:, 0] == i][:, 1], self.cond_k, replace=False))
                             if cond_proc_counts[i] > self.cond_k else torch.from_numpy(
            cond_proc_indices[cond_proc_indices[:, 0] == i][:, 1])
                             for i in range(len(cf_batch))]

        cond_drug_batch = self.cond_drug_matrix[cf_batch] if len(cf_batch) > 1 else np.expand_dims(
            self.cond_drug_matrix[cf_batch], axis=0)
        cond_drug_mask = cond_drug_batch > 0
        cond_drug_indices = np.argwhere(cond_drug_mask)
        cond_drug_counts = cond_drug_mask.sum(axis=1)
        cond_drug_sampled = [torch.from_numpy(
            np.random.choice(cond_drug_indices[cond_drug_indices[:, 0] == i][:, 1], self.cond_k, replace=False))
                             if cond_drug_counts[i] > self.cond_k else torch.from_numpy(
            cond_drug_indices[cond_drug_indices[:, 0] == i][:, 1])
                             for i in range(len(cf_batch))]
        # print("LEN,", len(cf_batch), len(cond_proc_sampled), len(cond_drug_sampled), len(positive_sampled), len(negative_sampled))
        cond_codes, proc_codes, drug_codes,pos_codes,neg_codes = self.drl_model.batch_encode(cf_batch, cond_proc_sampled, cond_drug_sampled,positive_sampled,negative_sampled) # enhance fea
        # print("AAAAAA", cond_code.shape, proc_code.shape, drug_code.shape) # B,; B,V, B,V
        batch_size = 256

        common_repr,rare_repr = [],[]
        for i in range(0, len(cf_batch), batch_size):
            cond_code = cond_codes[i:i+batch_size]
            proc_code = proc_codes[i:i+batch_size]
            drug_code = drug_codes[i:i+batch_size]
            # pos_code = pos_codes[i:i+batch_size]
            # neg_code = neg_codes[i:i+batch_size]
            cond_common_recon, cond_rare_recon = self.drl_model.ana(cond_code, proc_code, drug_code)
            # cond_common_recon, cond_rare_recon = cond_common_recon.squeeze(), cond_rare_recon.squeeze()
            common_repr.append(cond_common_recon)
            rare_repr.append(cond_rare_recon)
        common_repr = torch.cat(common_repr, dim=0)
        rare_repr = torch.cat(rare_repr, dim=0)
        # print("CCCCCCCCC", common_repr.shape)

        # cond_common_recon, cond_rare_recon = self.drl_model.predict(cond_code.unsqueeze(dim=0), proc_code.unsqueeze(dim=0), drug_code.unsqueeze(dim=0))
        return common_repr, rare_repr


    def get_drl_input(self, cond_code, cond_wc):
        # b,v,m = cond_code.size()
        # cond_code = cond_code.view(b*v, m)
        # cond_wc = cond_wc.view(b*v, m)
        cond_wc = cond_wc!=0 # 转为mask
        cf_batch = cond_code[cond_wc].cpu() # code lis
        if len(cf_batch) ==0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        # 处理ref_matrix
        rows = self.ref_matrix[cf_batch] if len(cf_batch)>1 else np.expand_dims(self.ref_matrix[cf_batch], axis=0)  # 获取batch中所有code对应的行
        positive_mask = rows > 0
        negative_mask = rows == 0
    
        # 使用numpy的argwhere和random.choice进行矢量化采样
        positive_indices = np.argwhere(positive_mask)
        positive_counts = positive_mask.sum(axis=1)

        positive_sampled = [torch.from_numpy(np.random.choice(positive_indices[positive_indices[:, 0] == i][:, 1], self.ref_k, replace=False))
             if positive_counts[i] > self.ref_k else torch.from_numpy(positive_indices[positive_indices[:, 0] == i][:, 1])
             for i in range(len(cf_batch))] # 如果没有则选择index为0的。

        negative_indices = np.argwhere(negative_mask)
        negative_counts = negative_mask.sum(axis=1)
        negative_sampled = [torch.from_numpy(np.random.choice(negative_indices[negative_indices[:, 0] == i][:, 1], self.ref_k, replace=False))
             if negative_counts[i] > self.ref_k else torch.from_numpy(negative_indices[negative_indices[:, 0] == i][:, 1])
             for i in range(len(cf_batch))]

        # 处理cond_proc_matrix和cond_drug_matrix，类似于ref_matrix
        cond_proc_batch = self.cond_proc_matrix[cf_batch] if len(cf_batch)>1 else np.expand_dims(self.cond_proc_matrix[cf_batch], axis=0)
        cond_proc_mask = cond_proc_batch > 0
        cond_proc_indices = np.argwhere(cond_proc_mask)
        cond_proc_counts = cond_proc_mask.sum(axis=1)
        cond_proc_sampled = [torch.from_numpy(np.random.choice(cond_proc_indices[cond_proc_indices[:, 0] == i][:, 1], self.cond_k, replace=False))
             if cond_proc_counts[i] > self.cond_k else torch.from_numpy(cond_proc_indices[cond_proc_indices[:, 0] == i][:, 1])
             for i in range(len(cf_batch))]

        cond_drug_batch = self.cond_drug_matrix[cf_batch]  if len(cf_batch)>1 else np.expand_dims(self.cond_drug_matrix[cf_batch], axis=0)
        cond_drug_mask = cond_drug_batch > 0
        cond_drug_indices = np.argwhere(cond_drug_mask)
        cond_drug_counts = cond_drug_mask.sum(axis=1)
        cond_drug_sampled = [torch.from_numpy(np.random.choice(cond_drug_indices[cond_drug_indices[:, 0] == i][:, 1], self.cond_k, replace=False))
             if cond_drug_counts[i] > self.cond_k else torch.from_numpy(cond_drug_indices[cond_drug_indices[:, 0] == i][:, 1])
             for i in range(len(cf_batch))]

        # 将结果转换为tensor并返回
        return cf_batch, cond_proc_sampled, cond_drug_sampled, positive_sampled, negative_sampled

    def obtain_drl_emb_joint(self, cond_code, proc_code, drug_code, cond_wc, cond_mask, pcf_model=None):
        '''joint train version，这里联合训练感觉可能会更好？'''

        diag_code, cond_proc, cond_drug, pos_diag, neg_diag = self.get_drl_input(cond_code, cond_wc)
        if len(diag_code) == 0: # diag没啥问题，
            drl_loss = {'loss':torch.tensor(0), 'task_aware_loss':torch.tensor(0)}
        else:
            drl_loss = self.drl_model.forward_joint(diag_code, cond_proc, cond_drug, pos_diag, neg_diag, pcf_model.embeddings) # 这里希望能感受到对比学习
        # print("BBBB",drl_loss['task_aware_loss']) # 替换的较少是吗
        cond_old_embedding , proc_new_emb, drug_new_emb = self.obtain_drl_emb(cond_code, proc_code, drug_code, cond_wc, cond_mask) # 用llm的embedding来着

        return drl_loss, cond_old_embedding, proc_new_emb, drug_new_emb


    def obtain_drl_emb(self, cond_code, proc_code, drug_code, cond_wc, cond_mask):
        """直接转为drl增强"""
        # reshape
        cond_old_emb = self.pcf_model.embeddings['conditions'](cond_code).detach() #强迫DRL自己学习
        proc_old_emb = self.pcf_model.embeddings['procedures'](proc_code).detach()
        drug_old_emb = self.pcf_model.embeddings['drugs'](drug_code).detach()
        # print("AAAAA", torch.isnan(cond_old_emb).any())
        # cond_common_recon, cond_rare_recon = self.drl_model.predict(cond_code, proc_code, drug_code) # enhance fea

        cond_common_recon, cond_rare_recon = self.drl_model.predict(cond_code, proc_code, drug_code, cond_old_emb, proc_old_emb, drug_old_emb) # enhance fea

        proc_new = self.drl_model.pcf_proc_cond_embedding(proc_code)
        drug_new = self.drl_model.pcf_drug_cond_embedding(drug_code)
        proc_new_emb = self.pcf_model.get_visit_emb(proc_new, feature_key='procedures', masks=proc_code!=0)
        drug_new_emb = self.pcf_model.get_visit_emb(drug_new, feature_key='drugs', masks=drug_code!=0)
        # print("AAABB", torch.isnan(cond_rare_recon).any())

        common_indices = (cond_wc == 0).nonzero(as_tuple=True)
        rare_indices = (cond_wc == 1).nonzero(as_tuple=True)

        # print(cond_wc.shape)
        # print(rare_indices[0].cpu().tolist())
        # print("common Num",len(common_indices[0]))
        # print("Patient_num, Change Num, Overall Num", len(set(rare_indices[0].cpu().tolist())), 16,len(rare_indices[0]), int(cond_mask.sum()))

        # print(common_indices.device, cond_old_emb.device, cond_common_recon.device)

        # 将 warm disease 的嵌入替换为新的嵌入
        # print("AAAAA",cond_old_emb[rare_indices][0]) # 看看第一个有啥差别
        cond_old_emb = cond_old_emb.clone() # 卧槽还真行，是对的，因为梯度传递到common_indices里面了
        cond_old_emb[common_indices] = cond_common_recon[common_indices] # common为啥不换，换了按理说会有text知识增强啊。直接inference的模式，换了会有很大影响
        cond_old_emb[rare_indices] = cond_rare_recon[rare_indices] # 用llm的embedding来着, 会不会是数据量太小了，影响就很小。
        cond_old_emb = cond_old_emb * cond_mask.unsqueeze(dim=-1) # mask, 因为common把padding也算进去了

        # print("BBBB",cond_rare_recon[rare_indices][0]-a)
        # print("-----------")


        cond_old_emb = self.pcf_model.get_visit_emb(cond_old_emb, feature_key='conditions', masks=cond_mask)# cond_old_emb.sum(dim=2) # B,V,H

        # print("XXXX", torch.isnan(cond_old_emb).any())
        return cond_old_emb, proc_new_emb, drug_new_emb
    
    def batch_encode(self,
                     batch: List[List[List[str]]],
                     padding: Tuple[bool, bool] = (True, True),
                     truncation: Tuple[bool, bool] = (True, True),
                     max_length: Tuple[int, int] = (10, 512),
                     ):
        voc = {'1.0': 1, '0.0': 0}
        if truncation[0]:
            batch = [tokens[-max_length[0] :] for tokens in batch]
        if truncation[1]:
            batch = [
                [tokens[-max_length[1] :] for tokens in visits] for visits in batch
            ]
        if padding[0]:
            batch_max_length = max([len(tokens) for tokens in batch])
            batch = [
                tokens + [["0.0"]] * (batch_max_length - len(tokens))
                for tokens in batch
            ]
        if padding[1]:
            batch_max_length = max(
                [max([len(tokens) for tokens in visits]) for visits in batch]
            )
            batch = [
                [
                    tokens + ["0.0"] * (batch_max_length - len(tokens))
                    for tokens in visits
                ]
                for visits in batch
            ]
        return [
            [[voc[token] for token in tokens] for tokens in visits]
            for visits in batch
        ]


    def forward(self,
        patient_id : List[List[str]],
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        drugs_hist: List[List[List[str]]],
        conditions_wc: List[List[List[str]]], # 0: common, 1: rare
        labels: List[List[str]],  # label
        **kwargs):
        # 需要和pcf保持一致
        # padding
        conditions_wc = self.batch_encode(conditions_wc)
        conditions_wc = torch.tensor(conditions_wc, dtype=torch.long, device=self.device)

        cond_code, proc_code, drug_code, cond_mask, proc_mask, drug_mask, condition_vis_emb, procedure_vis_emb, drug_history_vis_emb = self.pcf_model.obtain_code(patient_id, conditions, procedures, drugs_hist) # emb
        cond_mask = cond_code!=0 # 防止特殊定义

        pred_dic_raw = self.pcf_model.drl_forward(condition_vis_emb, procedure_vis_emb, drug_history_vis_emb, cond_mask, labels) # emb, 返回需要一摸一样
        condition_vis_emb_new , _, _= self.obtain_drl_emb(cond_code, proc_code, drug_code, conditions_wc, cond_mask)
        # condition_vis_emb = (condition_vis_emb + condition_vis_emb_new)/2 # 直接加和
        # print("Code num: {}, Change num: {}".format(cond_mask.sum(),conditions_wc.sum()))

        # pred_dic = self.pcf_model.drl_forward(self.drop(condition_vis_emb_new), self.drop(procedure_vis_emb), drug_history_vis_emb, cond_mask, labels)
        # pred_dic = self.pcf_model.drl_forward(self.drop2(condition_vis_emb_new+condition_vis_emb), self.drop2(procedure_vis_emb), drug_history_vis_emb, cond_mask, labels) # drugrec没有drop似乎
        pred_dic = self.pcf_model.drl_forward(condition_vis_emb_new, procedure_vis_emb, drug_history_vis_emb, cond_mask, labels) # drugrec没有drop似乎; eicu也没有，这种数量很少的code,影响比较大
        # pred_dic['y_prob'] = self.linear_label(pred_dic['y_prob'])

        pred_dic['y_prob'] =(pred_dic_raw['y_prob'] + pred_dic['y_prob'])/2 # ensemble

        return pred_dic

    def forward_s(self,
        patient_id : List[List[str]],
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        drugs_hist: List[List[List[str]]],
        conditions_wc: List[List[List[str]]], # 0: common, 1: rare
        labels: List[List[str]],  # label
        **kwargs):
        # 联合训练，这里对应提高tuning，设置epoch为0，
        # padding
        conditions_wc = self.batch_encode(conditions_wc)
        conditions_wc = torch.tensor(conditions_wc, dtype=torch.long, device=self.device)

        cond_code, proc_code, drug_code, cond_mask, proc_mask, drug_mask, condition_vis_emb, procedure_vis_emb, drug_history_vis_emb = self.pcf_model.obtain_code(patient_id, conditions, procedures, drugs_hist) # emb
        cond_mask = cond_code!=0 # 防止特殊定义

        pred_dic_raw = self.pcf_model.drl_forward(condition_vis_emb, procedure_vis_emb, drug_history_vis_emb, cond_mask, labels) # emb, 返回需要一摸一样
        loss_drl, condition_vis_emb_new,_, _ = self.obtain_drl_emb_joint(cond_code, proc_code, drug_code, conditions_wc, cond_mask, self.pcf_model)
        # condition_vis_emb = self.linear_fea(condition_vis_emb) #
        # print("Code num: {}, Change num: {}".format(cond_mask.sum(),conditions_wc.sum()))

        pred_dic = self.pcf_model.drl_forward(self.drop(condition_vis_emb_new), self.drop(procedure_vis_emb), drug_history_vis_emb, cond_mask, labels) # emb, 返回需要一摸一样
        # pred_dic['y_prob'] = self.linear_label(pred_dic['y_prob']) # label不太行

        pred_dic['y_prob'] = (pred_dic_raw['y_prob'] + pred_dic['y_prob'])/2 # 直接加和
        pred_dic['loss'] = 0.1* loss_drl['loss'] + pred_dic['loss'] # 重构损失也加上。
        # print("BBBB", loss_drl['loss'], pred_dic['loss'])
        return pred_dic # 为的是感知contrastive learning

# linear_label, 都使用pcf_model,linear_fea
