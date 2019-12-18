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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
if device != "cpu":
    torch.cuda.set_device(0)
# device = torch.device('cpu')
# torch.cuda.set_device(device)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True

class BertForJointShallowSemanticParsing(BertPreTrainedModel):
    def __init__(self, config, num_senses=2, num_args=2, lufrmap=None, frargmap=None, masking=True):
        super(BertForJointShallowSemanticParsing, self).__init__(config)
        self.masking = masking
        self.num_senses = num_senses # total number of all frames
        self.num_args = num_args # total number of all frame elements
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sense_classifier = nn.Linear(config.hidden_size, num_senses)
        self.arg_classifier = nn.Linear(config.hidden_size, num_args)
        self.lufrmap = lufrmap # mapping table for lu to its frame candidates    
        self.frargmap = frargmap # mapping table for lu to its frame candidates
        
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, lus=None, senses=None, args=None, using_gold_fame=False, position_ids=None, head_mask=None):
        
        torch.cuda.set_device(0)
#         torch.cuda.set_device(device)
        
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        sense_logits = self.sense_classifier(pooled_output)
        arg_logits = self.arg_classifier(sequence_output)
        
        # masking frame logits
        if self.masking:
            lufr_masks = utils.get_masks(lus, self.lufrmap, num_label=self.num_senses, masking=self.masking).to(device)        
            masked_sense_logits = sense_logits * lufr_masks
        else:
            masked_sense_logits = sense_logits
            
        # train frame identifier
        loss_fct_sense = CrossEntropyLoss()
        loss_sense = loss_fct_sense(masked_sense_logits.view(-1, self.num_senses), senses.view(-1))
        
        # masking arg logits
        pred_senses = []
        if senses is not None:
            for i in range(len(masked_sense_logits)):
                masked_sense_logit = masked_sense_logits[i]                    
                pred_sense, sense_score = utils.logit2label(masked_sense_logit)
                pred_senses.append(pred_sense)
        frarg_mask = utils.get_masks(pred_senses, self.frargmap, num_label=self.num_args, masking=True).to(device)
        
        frarg_mask = frarg_mask.view(len(frarg_mask), 1, -1)
        frarg_mask = frarg_mask.repeat(1, len(arg_logits[0]), 1)
        
        masked_arg_logits = arg_logits * frarg_mask
                
        # train arg classifier
        loss_fct_arg = CrossEntropyLoss()        
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = masked_arg_logits.view(-1, self.num_args)[active_loss]
            active_labels = args.view(-1)[active_loss]
            loss_arg = loss_fct_arg(active_logits, active_labels)
        else:
            loss_arg = loss_fct_arg(masked_arg_logits.view(-1, self.num_args), args.view(-1))
            
        # weighted sum of losses
        loss = 0.5*loss_sense + 0.5*loss_arg
#         print(loss_sense)
#         print(loss_arg)
#         print(loss)
#         print('')

        
        if senses is not None:
            return loss
        else:
            return sense_logits, arg_logits
