import itertools
import time
import pandas as pd
import torch
import dgl
import os
import math
import pkg_resources
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional, Union
from pyhealth.models.utils import get_last_visit
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.nn.functional import multilabel_margin_loss
from pyhealth.metrics import ddi_rate_score
from pyhealth.models.utils import batch_to_multihot
from pyhealth.models import BaseModel
from pyhealth.medcode import ATC
from pyhealth.datasets import SampleEHRDataset
from pyhealth import BASE_CACHE_PATH as CACHE_PATH
from config import config
from utils import pad_nested_list



class RecLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, voc_size, feature_num=2, dropout=0.1, nhead=2):
        super(RecLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.ddi_weight = 0.
        self.multiloss_weight = config['PCF_CONFIG']['MULTI']
        self.feature_num = feature_num
        self.bceloss = nn.BCELoss(reduction='none')


        self.rnns = torch.nn.TransformerEncoderLayer(
                    d_model=self.feature_num*embedding_dim, nhead=nhead, batch_first=True) # dropout 0.2
        # self.rnns = torch.nn.GRU(2*embedding_dim,2* embedding_dim, num_layers=1, batch_first=True, dropout=dropout)
        # self.rnns = nn.MultiheadAttention(2 * embedding_dim, num_heads=1, batch_first=True)
        # self.rnns
        self.id_proj = nn.Sequential(nn.Linear(self.feature_num * embedding_dim, self.feature_num*embedding_dim, bias=False),
                                     )
        self.fina_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear((self.feature_num+self.feature_num)*embedding_dim, voc_size))
        # self.fina_proj2 = nn.Sequential(nn.Dropout(dropout), nn.Linear((self.feature_num+self.feature_num)*embedding_dim, voc_size))
        if config['TASK'] == "DIAG":
            self.final_act = nn.Sigmoid()#, 和sigmoid差不多其实。看看对后续有啥影响吗
        else:
            self.final_act = nn.Sigmoid()

    def forward(
            self,
            patient_id: torch.Tensor,
            patient_emb: torch.Tensor,
            labels: torch.Tensor,
            ddi_adj: torch.Tensor,
            mask: torch.Tensor,
            labels_indexes: Optional[torch.Tensor] = None,
            # drl=False,
    ):
        patient_emb = self.rnns(patient_emb, src_key_padding_mask=~mask)
        # patient_emb, _ = self.rnns(patient_emb) # GRU
        # patient_emb, _ = self.rnns(patient_emb, patient_emb, patient_emb,
        #                                  key_padding_mask=~mask)  # B, V, 3D
        patient_emb = get_last_visit(patient_emb, mask) # B, 3D
        patient_id = self.id_proj(patient_id) # B, D, diag这个玩意很有作用
        patient_emb = torch.cat([patient_id, patient_emb], dim=1) # B, 4D
        # if drl:
        #     logits = self.fina_proj2(patient_emb) # 类似lora tuning
        # else:
        logits = self.fina_proj(patient_emb) # B, Label_size

        # y_prob = torch.sigmoid(logits)

        y_prob = self.final_act(logits)

        loss = self.calculate_loss(logits, y_prob, ddi_adj, labels, labels_indexes)
        return loss, y_prob


    
    def calculate_loss(
            self,
            logits: torch.Tensor,
            y_prob: torch.Tensor,
            ddi_adj: torch.Tensor,
            labels: torch.Tensor,
            label_index: Optional[torch.Tensor] = None,
    ):

        loss_cls = binary_cross_entropy_with_logits(logits, labels) #

        if self.multiloss_weight > 0 and label_index is not None:
            loss_multi = multilabel_margin_loss(y_prob, label_index)
            loss_cls = self.multiloss_weight * loss_multi + (1 - self.multiloss_weight) * loss_cls


        return loss_cls



class Transformer(BaseModel):
    def __init__(
            self,
            dataset: SampleEHRDataset,
            feature_keys=["conditions", "procedures", "drugs"],
            label_key="labels",
            mode="multilabel",

            # hyper related
            dropout: float = 0.3,
            num_rnn_layers: int = 2,
            embedding_dim: int = 64,
            hidden_dim: int = 64,
            **kwargs,
    ):
        super(Transformer, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        # define
        self.num_rnn_layers = num_rnn_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.dropout_id = torch.nn.Dropout(self.dropout)

        self.feat_tokenizers = self.get_feature_tokenizers() # tokenizer
        self.label_tokenizer = self.get_label_tokenizer() # 注意这里的drug可没有spec_token; 这里label索引需要加2对于正则化
        self.label_size = self.label_tokenizer.get_vocabulary_size()

        # save ddi adj
        self.ddi_adj = torch.nn.Parameter(self.generate_ddi_adj(), requires_grad=False)
        ddi_adj = self.generate_ddi_adj() # 用于存储
        np.save(os.path.join(CACHE_PATH, "ddi_adj.npy"), ddi_adj.numpy()) # 计算ddi直接从这里读取

        # module
        self.feature_keys_subs = ['conditions', 'procedures', 'drugs']
        self.rec_layer = RecLayer(self.embedding_dim, self.hidden_dim, self.label_size, feature_num=len(self.feature_keys_subs),  dropout=dropout)

        # 特殊/tmp定义， 不能放在init_weights之前，会使得embedding机制失效
        # init params
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)  # ehr emb


    def generate_ddi_adj(self) -> torch.FloatTensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True) # dataframe，这里使用了gamenet的ddi,不要存储
        # ddi = pd.read_csv('/home/czhaobo/KnowHealth/data/REC/MIII/processed/ddi_pairs.csv', header=0, index_col=0).values.tolist()
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = np.zeros((self.label_size, self.label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi # each row
        ]

        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        ddi_adj = torch.FloatTensor(ddi_adj)
        return ddi_adj

    def encode_patient(self, feature_key: str, raw_values: List[List[List[str]]]) -> torch.Tensor:
        """Encode patient data."""
        codes = self.feat_tokenizers[feature_key].batch_encode_3d(raw_values, max_length=[config['PCF_CONFIG']['MAXSEQ'],config['PCF_CONFIG']['MAXCODESEQ']]) # 这里会padding, B,V,M
        codes = torch.tensor(codes, dtype=torch.long, device=self.device)
        masks = codes!=0 # B,V,M
        embeddings = self.embeddings[feature_key](codes) # B,V,M,D
        embeddings = self.dropout_id(embeddings)
        visit_emb = self.get_visit_emb(embeddings) # B,V,D
        return codes, embeddings, masks, visit_emb # B,V, D

    def get_visit_emb(self, emb, feature_key=None, masks=None):
        """Get visit embedding."""
        return torch.sum(emb, dim=2)

    def decode_label(self, array_prob, tokenizer):
        """给定y概率，label tokenizer，返回所解码出来的code Token"""
        array_prob[array_prob >= config['PCF_CONFIG']['THRES']] = 1
        array_prob[array_prob < config['PCF_CONFIG']['THRES']] = 0 # 优化同步
        indices = [np.where(row == 1)[0].tolist() for row in array_prob]
        tokens = tokenizer.batch_decode_2d(indices)
        return tokens

    def obtain_code(
        self,
        patient_id : List[List[str]],
        conditions: List[List[List[str]]], # 需要和dataset保持一致[名字，因为**data]
        procedures: List[List[List[str]]],
        drugs_hist: List[List[List[str]]],
        # conditions_wc: List[List[List[str]]], # 需要和dataset保持一致[名字，因为**data]
    ):
        """获取embedding，用于后续的DRL"""
        cond_code, _, condi_mask, condition_vis_emb = self.encode_patient("conditions", conditions) # [B,V,M] [B,V,M,D]; [B,V,M], [B,V,D]
        proc_code, _, proc_mask, procedure_vis_emb = self.encode_patient("procedures", procedures)
        drug_code, _, drug_mask, drug_history_vis_emb = self.encode_patient("drugs", drugs_hist)

        return cond_code, proc_code, drug_code, condi_mask, proc_mask, drug_mask, condition_vis_emb, procedure_vis_emb, drug_history_vis_emb


    def drl_forward(
        self,
        condition_emb,
        procedure_emb,
        drugs_history_emb,
        cond_mask,
        labels: List[List[str]],  # label
        **kwargs,):
        """和obtain_code一起使用"""
        # prepare labels
        labels_index = self.label_tokenizer.batch_encode_2d(
            labels, padding=False, truncation=False
        ) # [[23,32],[1,2,3]]，注意比feature_tokenizer少两位

        labels = batch_to_multihot(labels_index, self.label_size) # tensor, B, Label_size;  # convert to multihot

        index_labels = -np.ones((len(labels), self.label_size), dtype=np.int64)
        for idx, cont in enumerate(labels_index):
            # remove redundant labels
            cont = list(set(cont))
            index_labels[idx, : len(cont)] = cont # remove padding and unk
        index_labels = torch.from_numpy(index_labels) # 类似的！【23，38，39】
        labels = labels.to(self.device) # for bce loss
        index_labels = index_labels.to(self.device) # for multi label loss

        mask = torch.sum(cond_mask, dim=-1) !=0 # visit-level mask; 这个更安全，emb相加可能为0
        patient_emb = torch.cat([condition_emb, procedure_emb, drugs_history_emb], dim=2) # B,V,3D;drugs_history_emb
        patient_id = torch.cat([self.dropout_id(get_last_visit(condition_emb, mask)), self.dropout_id(get_last_visit(procedure_emb,mask)), self.dropout_id(get_last_visit(drugs_history_emb,mask))],dim=1) # ,

        # patient_emb = torch.cat([condition_emb, procedure_emb], dim=2) # B,V,3D;drugs_history_emb for diag
        # patient_id = torch.cat([self.dropout_id(get_last_visit(condition_emb, mask)), self.dropout_id(get_last_visit(procedure_emb,mask))],dim=1) # ,

        # calculate loss
        loss, y_prob = self.rec_layer( # patient
            patient_id,
            patient_emb=patient_emb,
            labels=labels,
            ddi_adj=self.ddi_adj,
            mask=mask,
            labels_indexes=index_labels,
        )

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": labels,
        }


    def forward(
        self,
        patient_id : List[List[str]],
        conditions: List[List[List[str]]], # 需要和dataset保持一致[名字，因为**data]
        procedures: List[List[List[str]]],
        drugs_hist: List[List[List[str]]],
        labels: List[List[str]],  # label
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.
        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor of shape [patient, visit, num_labels]
                    representing the probability of each drug.
                y_true: a tensor of shape [patient, visit, num_labels]
                    representing the ground truth of each drug.
        """
        # # patient id
        # prepare labels
        labels_index = self.label_tokenizer.batch_encode_2d(
            labels, padding=False, truncation=False
        ) # [[23,32],[1,2,3]]，注意比feature_tokenizer少两位

        labels = batch_to_multihot(labels_index, self.label_size) # tensor, B, Label_size;  # convert to multihot

        index_labels = -np.ones((len(labels), self.label_size), dtype=np.int64)
        for idx, cont in enumerate(labels_index):
            # remove redundant labels
            cont = list(set(cont))
            index_labels[idx, : len(cont)] = cont # remove padding and unk
        index_labels = torch.from_numpy(index_labels) # 类似的！【23，38，39】
        labels = labels.to(self.device) # for bce loss
        index_labels = index_labels.to(self.device) # for multi label loss

        # patient id
        cond_code, _, condi_mask, condition_vis_emb = self.encode_patient("conditions", conditions) # [B,V,M] [B,V,M,D]; [B,V,M], [B,V,D]
        proc_code, _, proc_mask, procedure_vis_emb = self.encode_patient("procedures", procedures)
        drug_code, _, drug_mask, drug_history_vis_emb = self.encode_patient("drugs", drugs_hist) # drug rec的时候不能放drug 1，1，1，1

        seq_emb = {'conditions': condition_vis_emb, 'procedures': procedure_vis_emb, 'drugs': drug_history_vis_emb}
        mask = torch.sum(condi_mask, dim=-1) !=0 # visit-level mask; 这个更安全，emb相加可能为0

        patient_emb = torch.cat([seq_emb[feature] for feature in self.feature_keys_subs], dim=2) # B,V,3D
        patient_id = torch.cat([self.dropout_id(get_last_visit(seq_emb[feature], mask)) for feature in self.feature_keys_subs],dim=1)

        # calculate loss
        loss, y_prob = self.rec_layer( # patient
            patient_id,
            patient_emb=patient_emb,
            labels=labels,
            ddi_adj=self.ddi_adj,
            mask=mask,
            labels_indexes=index_labels,
        )


        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": labels,
        }


