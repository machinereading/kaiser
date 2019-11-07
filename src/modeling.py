from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../../')
from kaiser.src import dataio
from kaiser.src import utils
from torch.nn.parameter import Parameter
from transformers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

class BertForJointShallowSemanticParsing(BertPreTrainedModel):
    def __init__(self, config, num_senses=2, num_args=2, lufrmap=None, frargmap=None, srl='framenet', masking=True):
        super(BertForJointShallowSemanticParsing, self).__init__(config)
        self.masking = masking
        self.num_senses = num_senses # total number of all frames
        self.num_args = num_args # total number of all frames
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sense_classifier = nn.Linear(config.hidden_size, num_senses)
        self.arg_classifier = nn.Linear(config.hidden_size, num_args)
        self.lufrmap = lufrmap # mapping table for lu to its frame candidates    
        self.frargmap = frargmap # mapping table for lu to its frame candidates
        
        self.init_weights()
        
#     # logit 2 label
#     def logit2label(self, logit, mask):
#         masking = np.multiply(logit, mask)
#         masking[masking==0] = np.NINF
#         sm = nn.Softmax()
#         pred_logits = sm(masking).view(1,-1)
#         score, label = pred_logits.max(1)
#         score = float(score)
#         return label, score

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, lus=None, senses=None, args=None, using_gold_fame=False, position_ids=None, head_mask=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        sense_logits = self.sense_classifier(pooled_output)
        arg_logits = self.arg_classifier(sequence_output)
        
        lufr_masks = utils.get_masks(lus, self.lufrmap, num_label=self.num_senses, masking=self.masking).to(device)
        
        sense_loss = 0 # loss for sense id
        arg_loss = 0 # loss for arg id
        if senses is not None:
            for i in range(len(sense_logits)):
                sense_logit = sense_logits[i]
                arg_logit = arg_logits[i]
                lufr_mask = lufr_masks[i]
                gold_sense = senses[i]
                gold_arg = args[i]
                
                #train sense classifier
                loss_fct_sense = CrossEntropyLoss(weight = lufr_mask)
                loss_per_seq_for_sense = loss_fct_sense(sense_logit.view(-1, self.num_senses), gold_sense.view(-1))
                sense_loss += loss_per_seq_for_sense
                
                #train arg classifier
                pred_sense, sense_score = utils.logit2label(sense_logit, lufr_mask)
                frarg_mask = utils.get_masks([pred_sense], self.frargmap, num_label=self.num_args, masking=True).to(device)[0]
                
                loss_fct_arg = CrossEntropyLoss(weight = frarg_mask)
                
                # only keep active parts of loss
                if attention_mask is not None:
                    active_loss = attention_mask[i].view(-1) == 1
                    active_logits = arg_logit.view(-1, self.num_args)[active_loss]
                    active_labels = gold_arg.view(-1)[active_loss]
                    loss_per_seq_for_arg = loss_fct_arg(active_logits, active_labels)
                else:
                    loss_per_seq_for_arg = loss_fct_arg(arg_logit.view(-1, self.num_args), gold_arg.view(-1))
                arg_loss += loss_per_seq_for_arg
            
            # 0.5 weighted loss
            total_loss = 0.5*sense_loss + 0.5*arg_loss
            loss = total_loss / len(sense_logits)
            return loss
        else:
            return sense_logits, arg_logits