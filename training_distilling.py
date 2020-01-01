
# coding: utf-8

# In[2]:


import sys
import glob
import torch
sys.path.append('../')
import os
from transformers import *
from kaiser.src import utils
from kaiser.src import dataio
from kaiser.src.modeling import BertForJointShallowSemanticParsing
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score, precision_score, recall_score

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
if device != "cpu":
    torch.cuda.set_device(0)
# device = torch.device('cpu')
# torch.cuda.set_device(device)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True

import numpy as np
import random
np.random.seed(0)   
random.seed(0)
import random

from torch import autograd
torch.cuda.empty_cache()


# In[3]:


KD_loss = nn.KLDivLoss(reduction='batchmean')


# In[3]:


# 실행시간 측정 함수
import time

_start_time = time.time()

def tic():
    global _start_time 
    _start_time = time.time()

def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60)
    
    result = '{}hour:{}min:{}sec'.format(t_hour,t_min,t_sec)
    return result


# In[4]:


try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'


# In[5]:


def train(teacher_dir=False, PRETRAINED_MODEL="bert-base-multilingual-cased", 
          temperature=2.0, alpha_distilling=0.5, alpha_parsing=0.5, 
          model_dir=False, epoch=50):

    print('teacher model:', teacher_dir)
    print('original model:', 'BERT-multilingual-base')
    print('\n\tyour model would be saved at', model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    teacher = BertForJointShallowSemanticParsing.from_pretrained(teacher_dir, 
                                                                   num_senses = len(bert_io.sense2idx), 
                                                                   num_args = len(bert_io.bio_arg2idx),
                                                                   lufrmap=bert_io.lufrmap, 
                                                                   frargmap = bert_io.bio_frargmap)

    student = BertForJointShallowSemanticParsing.from_pretrained(PRETRAINED_MODEL, 
                                                               num_senses = len(bert_io.sense2idx), 
                                                               num_args = len(bert_io.bio_arg2idx),
                                                               lufrmap=bert_io.lufrmap, 
                                                               frargmap = bert_io.bio_frargmap)
    teacher.to(device)
    student.to(device)  

    tic()
    print('\n### converting data to BERT input...')
    trn_data = bert_io.convert_to_bert_input_JointShallowSemanticParsing(trn)
    print('\t ...is done:', tac())
    print('\t#of instance:', len(trn), len(trn_data))
    sampler = RandomSampler(trn)
    trn_dataloader = DataLoader(trn_data, sampler=sampler, batch_size=batch_size)
    
    # load optimizer
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(student.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(student.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
    
        
    max_grad_norm = 1.0
    num_of_epoch = 0
    accuracy_result = []
    
    for _ in trange(epochs, desc="Epoch"):
        
        # TRAIN loop
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(trn_dataloader):
            
            # add batch to gpu
            torch.cuda.set_device(0)
#             torch.cuda.set_device(device)
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_orig_tok_to_maps, b_input_lus, b_input_senses, b_input_args, b_token_type_ids, b_input_masks = batch            
            
            teacher.eval()
            student.train()
            
            with torch.no_grad():
                t_frame_logits, t_arg_logits = teacher(b_input_ids, lus=b_input_lus,
                                 token_type_ids=b_token_type_ids, attention_mask=b_input_masks)
                
            s_frame_logits, s_arg_logits = student(b_input_ids, lus=b_input_lus,
                             token_type_ids=b_token_type_ids, attention_mask=b_input_masks)
                        
            loss_distilling_frame = (
                KD_loss(
                    F.log_softmax(s_frame_logits / temperature, dim=-1),
                    F.softmax(t_frame_logits / temperature, dim=-1),            
                )
                * temperature **2
            )
            
            loss_distilling_arg = (
                KD_loss(
                    F.log_softmax(s_arg_logits / temperature, dim=-1),
                    F.softmax(t_arg_logits / temperature, dim=-1),            
                )
                * temperature **2
            )
            
            loss_distilling = 0.5*loss_distilling_frame + 0.5*loss_distilling_arg
            loss = alpha_distilling * loss_distilling
            
            loss_fct_frame = CrossEntropyLoss()
            loss_frame = loss_fct_frame(s_frame_logits.view(-1, len(bert_io.sense2idx)), b_input_senses.view(-1))
            
            loss_fct_arg = CrossEntropyLoss()
            active_loss = b_input_masks.view(-1) == 1
            active_logits = s_arg_logits.view(-1, len(bert_io.bio_arg2idx))[active_loss]
            active_labels = b_input_args.view(-1)[active_loss]
            loss_arg = loss_fct_arg(active_logits, active_labels)
            
            loss_parsing = 0.5*loss_frame + 0.5*loss_arg
                                              
            loss += alpha_parsing * loss_parsing

            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=student.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            student.zero_grad()

        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        
        # save your model
        model_saved_path = model_dir+str(num_of_epoch)+'/'
        print('\n\tyour model is saved:', model_saved_path)
        if not os.path.exists(model_saved_path):
            os.makedirs(model_saved_path)
        student.save_pretrained(model_saved_path)
        
        num_of_epoch += 1

        
#         break
    print('...training is done')


# In[6]:


srl = 'framenet'
masking = True
MAX_LEN = 256
batch_size = 3
PRETRAINED_MODEL = "bert-base-multilingual-cased"
fnversion = '1.7'
language = 'multi'


# In[ ]:


PRETRAINED_MODEL = "bert-base-multilingual-cased"
teacher_dir='/disk/data/models/frameBERT/frameBERT_en/'
temperature=2.0
alpha_ce=0.5
alpha_frame=0.5


# In[ ]:


model_dir = '/disk/data/models/framenet/distilling/'
epochs = 50
trn, dev, tst = dataio.load_data(srl=srl, language='ko')
language = 'multi'

print('')
print('### TRAINING')
print('MODEL:', srl)
print('LANGUAGE:', language)
print('PRETRAINED BERT:', PRETRAINED_MODEL)
print('training data:')
print('\t(ko):', len(trn))
print('BATCH_SIZE:', batch_size)
print('MAX_LEN:', MAX_LEN)
print('')

bert_io = utils.for_BERT(mode='train', srl=srl, language=language, masking=masking, fnversion=fnversion, pretrained=PRETRAINED_MODEL)
train(teacher_dir=teacher_dir,
     temperature=2.0, alpha_distilling=0.5, alpha_parsing=0.5, 
      model_dir=model_dir, epoch=epochs)

