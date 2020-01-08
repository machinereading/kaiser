
# coding: utf-8

# In[1]:


import json
import sys
import glob
import torch
sys.path.append('../')
import os
from transformers import *
from kaiser.src import utils
from kaiser.src import dataio
from kaiser.src.modeling import BertForJointShallowSemanticParsing, FrameBERT
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


# In[2]:


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


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


# In[6]:


with open('./koreanframenet/resource/info/fn1.7_frame2idx.json', 'r') as f:
    frame2idx = json.load(f)
with open('./koreanframenet/resource/info/fn1.7_frame_definitions.json', 'r') as f:
    frame2definition = json.load(f)
frame_prototype = torch.load('./data/frame_prototype.pt')
    

def get_prototype(input_senses):

    frame_prototypes = []
    for i in input_senses:
        frame_idx = i.item()
        frame = bert_io.idx2sense[frame_idx]
    
        proto = frame_prototype[frame]
        frame_prototypes.append(proto)

    frame_prototypes = torch.stack(frame_prototypes).view(-1, 768)
    
    return frame_prototypes


# In[7]:


def train(PRETRAINED_MODEL="bert-base-multilingual-cased", 
          temperature=2.0, alpha_distilling=0.2, alpha_parsing=0.8, 
          model_dir=False, epoch=50):

    print('original model:', 'BERT-multilingual-base')
    print('\n\tyour model would be saved at', model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # MLP for frame prototype
    model_path = '/disk/data/models/framenet/prototype_mlp/prototype_mlp.pth'
    mlp_model = MLP()
    mlp_model.to(device)
    mlp_model.load_state_dict(torch.load(model_path))
    mlp_model.eval()
    
    # FrameBERT_ko model
    frameBERT_dir = '/disk/data/models/frameBERT/frameBERT_en'
    model = BertForJointShallowSemanticParsing.from_pretrained(frameBERT_dir, 
                                                               num_senses = len(bert_io.sense2idx), 
                                                               num_args = len(bert_io.bio_arg2idx),
                                                               lufrmap=bert_io.lufrmap, 
                                                               frargmap = bert_io.bio_frargmap, 
                                                               return_pooled_output=True)
    
    model.to(device)  

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
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters()) 
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
            mlp_model.eval()
            model.train()
            
            # add batch to gpu
            torch.cuda.set_device(0)
#             torch.cuda.set_device(device)
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_orig_tok_to_maps, b_input_lus, b_input_senses, b_input_args, b_token_type_ids, b_input_masks = batch                      
            
            # get prototypes for frames
            prototypes = get_prototype(b_input_senses)
            
            pooled_output, loss_parsing = model(b_input_ids, 
                                                token_type_ids=b_token_type_ids, 
                                                attention_mask=b_input_masks,
                                                lus=b_input_lus,
                                                senses=b_input_senses, 
                                                args=b_input_args)
            
            with torch.no_grad():
                pooled_output = mlp_model(pooled_output)
            
            loss_distilling = (
                KD_loss(
                    F.log_softmax(pooled_output / temperature, dim=-1),
                    F.softmax(prototypes / temperature, dim=-1),            
                )
                * temperature **2
            )
            
            loss = alpha_distilling * loss_distilling + alpha_parsing * loss_parsing
            
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
#             scheduler.step()
            model.zero_grad()
    
#             break

        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        
#         break
#         save your model
        model_saved_path = model_dir+str(num_of_epoch)+'/'
        print('\n\tyour model is saved:', model_saved_path)
        if not os.path.exists(model_saved_path):
            os.makedirs(model_saved_path)
        model.save_pretrained(model_saved_path)
        
        num_of_epoch += 1

        
#         break
    print('...training is done')


# In[9]:


def train_ori(retrain=False, pretrained_dir=False):
    if pretrained_dir:
        print('original model:', pretrained_dir)
    else:
        print('original model:', 'BERT-multilingual-base')
    print('\n\tyour model would be saved at', model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # load a model first
    if retrain:
#         model_saved_path = pretrained_dir
        model = BertForJointShallowSemanticParsing.from_pretrained(pretrained_dir, 
                                                                   num_senses = len(bert_io.sense2idx), 
                                                                   num_args = len(bert_io.bio_arg2idx),
                                                                   lufrmap=bert_io.lufrmap, 
                                                                   frargmap = bert_io.bio_frargmap)
    else:
#         model_saved_path = PRETRAINED_MODEL
        model = BertForJointShallowSemanticParsing.from_pretrained(PRETRAINED_MODEL, 
                                                                   num_senses = len(bert_io.sense2idx), 
                                                                   num_args = len(bert_io.bio_arg2idx),
                                                                   lufrmap=bert_io.lufrmap, 
                                                                   frargmap = bert_io.bio_frargmap)
    model.to(device)
    
    print('retrain:', retrain)
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
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

#     lr = 5e-5
#     lr =3e-5
#     optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
#     num_training_steps = len(trn_dataloader) // epochs
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)  # PyTorch scheduler
    
    max_grad_norm = 1.0
#     global_step = 0
#     num_of_epoch = 1
    num_of_epoch = 0
    accuracy_result = []
    for _ in trange(epochs, desc="Epoch"):
        
        # TRAIN loop
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(trn_dataloader):
            model.train()
            # add batch to gpu
            torch.cuda.set_device(0)
#             torch.cuda.set_device(device)
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_orig_tok_to_maps, b_input_lus, b_input_senses, b_input_args, b_token_type_ids, b_input_masks = batch            
#             print(b_token_type_ids[0])
            # forward pass
#             with autograd.detect_anomaly():
            loss = model(b_input_ids, lus=b_input_lus, senses=b_input_senses, args=b_input_args,
                     token_type_ids=b_token_type_ids, attention_mask=b_input_masks)
            # backward pass


            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
#             scheduler.step()
            model.zero_grad()
    
#             break

        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        
        # save your model
        model_saved_path = model_dir+str(num_of_epoch)+'/'
        print('\n\tyour model is saved:', model_saved_path)
        if not os.path.exists(model_saved_path):
            os.makedirs(model_saved_path)
        model.save_pretrained(model_saved_path)
        
        num_of_epoch += 1

        
#         break
    print('...training is done')


# # Train 10%

# In[ ]:


srl = 'framenet'
masking = True
MAX_LEN = 256
batch_size = 6
PRETRAINED_MODEL = "bert-base-multilingual-cased"
fnversion = '1.7'
language = 'multi'


# In[10]:


trn, dev, tst = dataio.load_data(srl=srl, language='ko')

# by 10% (1783)
trn = random.sample(trn, k=1783)

with open('./trn_10per.json','w') as f:
    json.dump(trn, f, ensure_ascii=False, indent=4)

# # by 25% (4460)
# trn = random.sample(trn, k=4460)

# # by 50% (8919)
# trn = random.sample(trn, k=8919)

print('\n\t### training instance:', len(trn))

epochs = 50
language = 'multi'


# ## for distill model (10%)

# In[1]:


model_dir = '/disk/data/models/framenet/proto_distilling-10/'

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
train(temperature=2.0, alpha_distilling=0.5, alpha_parsing=0.5, 
      model_dir=model_dir, epoch=epochs)


# ## for koModel (10%)

# In[2]:


# for KoModel

model_dir = '/disk/data/models/framenet/koModel-10/'

print('\nFineTuning Multilingual')
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
train_ori()


# ## for mul-Model (10%)

# In[3]:


model_dir = '/disk/data/models/framenet/mulModel-10/'

print('\nFineTuning FrameBERT')
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
train_ori(retrain=True, pretrained_dir='/disk/data/models/frameBERT/frameBERT_en/')


# # Train 25%

# In[ ]:


trn, dev, tst = dataio.load_data(srl=srl, language='ko')

# by 10% (1783)
# trn = random.sample(trn, k=1783)

# by 25% (4460)
trn = random.sample(trn, k=4460)

with open('./trn_25per.json','w') as f:
    json.dump(trn, f, ensure_ascii=False, indent=4)
    
# # by 50% (8919)
# trn = random.sample(trn, k=8919)

print('\n\t### training instance:', len(trn))

epochs = 50
language = 'multi'


# ## for distill model (25%)

# In[ ]:


model_dir = '/disk/data/models/framenet/proto_distilling-25/'

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
train(temperature=2.0, alpha_distilling=0.5, alpha_parsing=0.5, 
      model_dir=model_dir, epoch=epochs)


# ## for koModel (25%)

# In[ ]:


# for KoModel

model_dir = '/disk/data/models/framenet/koModel-25/'

print('\nFineTuning Multilingual')
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
train_ori()


# ## for mul-Model (25%)

# In[ ]:


model_dir = '/disk/data/models/framenet/mulModel-25/'

print('\nFineTuning FrameBERT')
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
train_ori(retrain=True, pretrained_dir='/disk/data/models/frameBERT/frameBERT_en/')


# # Train 50%

# In[ ]:


# trn, dev, tst = dataio.load_data(srl=srl, language='ko')

# # by 10% (1783)
# # trn = random.sample(trn, k=1783)

# # # by 25% (4460)
# # trn = random.sample(trn, k=4460)

# # by 50% (8919)
# trn = random.sample(trn, k=8919)

# print('\n\t### training instance:', len(trn))

# epochs = 50
# # epochs = 5
# language = 'multi'


# ## for koModel

# In[ ]:


# # for KoModel

# model_dir = '/disk/data/models/framenet/koModel-50/'

# print('\nFineTuning Multilingual')
# print('### TRAINING')
# print('MODEL:', srl)
# print('LANGUAGE:', language)
# print('PRETRAINED BERT:', PRETRAINED_MODEL)
# print('training data:')
# print('\t(ko):', len(trn))
# print('BATCH_SIZE:', batch_size)
# print('MAX_LEN:', MAX_LEN)
# print('')

# bert_io = utils.for_BERT(mode='train', srl=srl, language=language, masking=masking, fnversion=fnversion, pretrained=PRETRAINED_MODEL)
# train_ori()


# ## for mul-Model

# In[ ]:


# model_dir = '/disk/data/models/framenet/mulModel-50/'

# print('\nFineTuning FrameBERT')
# print('### TRAINING')
# print('MODEL:', srl)
# print('LANGUAGE:', language)
# print('PRETRAINED BERT:', PRETRAINED_MODEL)
# print('training data:')
# print('\t(ko):', len(trn))
# print('BATCH_SIZE:', batch_size)
# print('MAX_LEN:', MAX_LEN)
# print('')

# bert_io = utils.for_BERT(mode='train', srl=srl, language=language, masking=masking, fnversion=fnversion, pretrained=PRETRAINED_MODEL)
# train_ori(retrain=True, pretrained_dir='/disk/data/models/frameBERT/frameBERT_en/')

