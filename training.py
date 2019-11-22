
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


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


def train():
    print('your model would be saved at', model_dir)
    
    if srl == 'propbank-dp':
        model = BertForJointShallowSemanticParsing.from_pretrained(PRETRAINED_MODEL, 
                                                                   num_senses = len(bert_io.sense2idx), 
                                                                   num_args = len(bert_io.arg2idx),
                                                                   srl=srl,
                                                                   masking=False)
    else:
        model = BertForJointShallowSemanticParsing.from_pretrained(PRETRAINED_MODEL, 
                                                                   num_senses = len(bert_io.sense2idx), 
                                                                   num_args = len(bert_io.bio_arg2idx),
                                                                   lufrmap=bert_io.lufrmap, 
                                                                   frargmap = bert_io.bio_frargmap)
    model.to(device);
    
    trn_data = bert_io.convert_to_bert_input_JointShallowSemanticParsing(trn)
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
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(trn_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_orig_tok_to_maps, b_input_lus, b_input_senses, b_input_args, b_input_masks = batch            
            # forward pass
            loss = model(b_input_ids, token_type_ids=None, lus=b_input_lus, senses=b_input_senses, args=b_input_args,
                         attention_mask=b_input_masks)
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
            model.zero_grad()
            
#             break

        # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    model_saved_path = model_dir+'epoch-'+str(num_of_epoch)+'-joint.pt'        
    torch.save(model, model_saved_path)
    num_of_epoch += 1
        
#         break
    print('...training is done')


# In[7]:


# srl = 'propbank-dp'
# language = 'ko'
# masking = False
# model_dir = '/disk/data/models/ko-srl-tgt-1117/'
# if language == 'en':
#     fnversion = 1.7
# #     PRETRAINED_MODEL = "bert-large-cased"
#     MAX_LEN = 256
#     batch_size = 6
#     PRETRAINED_MODEL = "bert-base-multilingual-cased"
# else:
#     fnversion = 1.1
#     PRETRAINED_MODEL = "bert-base-multilingual-cased"
#     MAX_LEN = 256
#     batch_size = 6

# epochs = 50

# trn, dev, tst = dataio.load_data(srl=srl, language=language)
# print('')
# print('MODEL:', srl)
# print('LANGUAGE:', language)

# bert_io = utils.for_BERT(mode='train', srl=srl, language=language, masking=masking, fnversion=fnversion)

# train()


# In[8]:


# srl = 'framenet'
# language = 'ko'
# masking = True
# model_dir = '/disk/data/models/ko-framenet-tgt-1117/'
# if language == 'en':
#     fnversion = 1.7
# #     PRETRAINED_MODEL = "bert-large-cased"
#     MAX_LEN = 256
#     batch_size = 6
#     PRETRAINED_MODEL = "bert-base-multilingual-cased"
# else:
#     fnversion = 1.1
#     PRETRAINED_MODEL = "bert-base-multilingual-cased"
#     MAX_LEN = 256
#     batch_size = 6

# epochs = 50

# trn, dev, tst = dataio.load_data(srl=srl, language=language)
# print('')
# print('MODEL:', srl)
# print('LANGUAGE:', language)

# bert_io = utils.for_BERT(mode='train', srl=srl, language=language, masking=masking, fnversion=fnversion)

# train()


# In[9]:


# srl = 'framenet'
# language = 'en'
# masking = True
# model_dir = '/disk/data/models/en-framenet-tgt-1117/'
# if language == 'en':
#     fnversion = 1.7
# #     PRETRAINED_MODEL = "bert-large-cased"
#     MAX_LEN = 256
#     batch_size = 6
#     PRETRAINED_MODEL = "bert-base-multilingual-cased"
# else:
#     fnversion = 1.1
#     PRETRAINED_MODEL = "bert-base-multilingual-cased"
#     MAX_LEN = 256
#     batch_size = 6

# epochs = 50

# trn, dev, tst = dataio.load_data(srl=srl, language=language)
# print('')
# print('MODEL:', srl)
# print('LANGUAGE:', language)

# bert_io = utils.for_BERT(mode='train', srl=srl, language=language, masking=masking, fnversion=fnversion)

# train()


# In[ ]:


srl = 'framenet'
language = 'en'
masking = True
model_dir = '/disk/data/models/en-framenet-tgt-large/'
if language == 'en':
    fnversion = 1.7
    PRETRAINED_MODEL = 'bert-large-cased'
#     PRETRAINED_MODEL = 'bert-large-cased-whole-word-masking'
    MAX_LEN = 256
    batch_size = 1
#     PRETRAINED_MODEL = "bert-base-multilingual-cased"
else:
    fnversion = 1.1
    PRETRAINED_MODEL = "bert-base-multilingual-cased"
    MAX_LEN = 256
    batch_size = 6

epochs = 16

trn, dev, tst = dataio.load_data(srl=srl, language=language)
print('')
print('MODEL:', srl)
print('LANGUAGE:', language)

bert_io = utils.for_BERT(mode='train', srl=srl, language=language, masking=masking, fnversion=fnversion, pretrained=PRETRAINED_MODEL)

train()

