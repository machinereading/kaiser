
# coding: utf-8

# In[1]:


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

from torch import autograd
torch.cuda.empty_cache()


# In[2]:


import random


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


def train(retrain=False, pretrained_dir=False):
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
        # load a fine-tuned model
#         print('epoch:', num_of_epoch)
#         print('epoch-1 model:', model_saved_path)
#         model = BertForJointShallowSemanticParsing.from_pretrained(model_saved_path, 
#                                                                    num_senses = len(bert_io.sense2idx), 
#                                                                    num_args = len(bert_io.bio_arg2idx),
#                                                                    lufrmap=bert_io.lufrmap, 
#                                                                    frargmap = bert_io.bio_frargmap)
#         model.to(device)
        
#         lr = 5e-5
#         optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
#         num_training_steps = len(trn_dataloader) // epochs
#         scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)  # PyTorch scheduler
        
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
        
        # load a fine-tuned model
#         model = BertForJointShallowSemanticParsing.from_pretrained(model_saved_path, 
#                                                                    num_senses = len(bert_io.sense2idx), 
#                                                                    num_args = len(bert_io.bio_arg2idx),
#                                                                    lufrmap=bert_io.lufrmap, 
#                                                                    frargmap = bert_io.bio_frargmap)
#         model.to(device)
        
        num_of_epoch += 1

        
#         break
    print('...training is done')


# In[6]:


srl = 'framenet'
masking = True
MAX_LEN = 256
batch_size = 6
PRETRAINED_MODEL = "bert-base-multilingual-cased"
fnversion = '1.7'
language = 'multi'


# # (1) train En-FN with exemplars model

# In[7]:


# model_dir = '/disk/data/models/framenet/enModel-with-exemplar/'
# trn, dev, tst = dataio.load_data(srl=srl, language='en')
# # trn = random.sample(trn, k=50)
# # epochs = 10
# epochs = 20

# print('')
# print('### TRAINING')
# print('MODEL:', srl)
# print('LANGUAGE:', language)
# print('PRETRAINED BERT:', PRETRAINED_MODEL)
# print('training data:')
# print('\t(en):', len(trn))
# print('BATCH_SIZE:', batch_size)
# print('MAX_LEN:', MAX_LEN)
# print('')

# bert_io = utils.for_BERT(mode='train', srl=srl, language=language, masking=masking, fnversion=fnversion, pretrained=PRETRAINED_MODEL)
# train()
# train(retrain=True, pretrained_dir='/disk/data/models/framenet/enModel-with-exemplar/0/')


# # (2) train KFN

# In[8]:


# model_dir = '/disk/data/models/framenet/koModel/'
# trn, dev, tst = dataio.load_data(srl=srl, language='ko')
# # trn = random.sample(trn, k=50)
# epochs = 50

# print('\nFrameBERT(ko)')
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
# train()


# # (3) fine-tuning by Korean FrameNet

# In[9]:


# by 100%

model_dir = '/disk/data/models/dict_framenet/mulModel-100/'
trn, dev, tst = dataio.load_data(srl=srl, language='ko')
# trn = random.sample(trn, k=20)
epochs = 50


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
train(retrain=True, pretrained_dir='/disk/data/models/dict_framenet/enModel-with-exemplar/9/')


# In[ ]:


# by 25% (4460)

# model_dir = '/disk/data/models/framenet/mulModel-25/'
# epochs = 50

# trn, dev, tst = dataio.load_data(srl=srl, language='ko')

# # trn = random.sample(trn, k=4460)
# trn = random.sample(trn, k=50)
# language = 'multi'

# print('')
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
# # train(retrain=True, pretrained_dir='/disk/data/models/framenet/enModel-with-exemplar/0/')


# In[ ]:


# by 50% (8919)

# model_dir = '/disk/data/models/framenet/mulModel-50/'
# epochs = 20

# trn, dev, tst = dataio.load_data(srl=srl, language='ko')

# trn = random.sample(trn, k=8919)
# language = 'multi'

# print('')
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
# train(retrain=True, pretrained_dir='/disk/data/models/dict_framenet/enModel-with-exemplar/6/')


# In[ ]:


# by 75% (13378)

# model_dir = '/disk/data/models/framenet/mulModel-75/'
# epochs = 20

# trn, dev, tst = dataio.load_data(srl=srl, language='ko')

# trn = random.sample(trn, k=13378)
# language = 'multi'

# print('')
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
# train(retrain=True, pretrained_dir='/disk/data/models/dict_framenet/enModel-with-exemplar/6/')

