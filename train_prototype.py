
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
from kaiser.src.prototypical_loss import prototypical_loss as loss_fn
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score, precision_score, recall_score

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from kaiser.src.prototypical_loss import prototypical_loss as loss_fn
from kaiser.src import prototypical_batch_sampler

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

from collections import Counter, OrderedDict


# In[2]:


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


# In[3]:


try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'


# In[4]:


bert_io = utils.for_BERT(mode='train', language='multi')


# In[5]:


frameBERT_dir = '/disk/data/models/frameBERT/frameBERT_en'

frameBERT = FrameBERT.from_pretrained(frameBERT_dir,
                                      num_senses = len(bert_io.sense2idx), 
                                      num_args = len(bert_io.bio_arg2idx),
                                      lufrmap=bert_io.lufrmap, 
                                      frargmap = bert_io.bio_frargmap)
frameBERT.to(device)
frameBERT.eval()


# In[6]:


print('... loading FN data')
tic()
trn, dev, tst = dataio.load_data(srl='framenet', language='en', exem=True)
# trn = random.sample(trn, k=500)
# dev = random.sample(trn, k=100)
# tst = random.sample(tst, k=100)
print('... converting FN data to BERT')
trn_data = bert_io.convert_to_bert_input_JointShallowSemanticParsing(trn)
dev_data = bert_io.convert_to_bert_input_JointShallowSemanticParsing(dev)
tst_data = bert_io.convert_to_bert_input_JointShallowSemanticParsing(tst)

with open('./koreanframenet/resource/info/fn1.7_frame2idx.json', 'r') as f:
    frame2idx = json.load(f)
with open('./koreanframenet/resource/info/fn1.7_frame_definitions.json', 'r') as f:
    frame2definition = json.load(f)

def_data, def_y = bert_io.convert_to_bert_input_label_definition(frame2definition, frame2idx)
print(tac())


# In[7]:


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


# In[8]:


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

def get_y(data):
    with open('./koreanframenet/resource/info/fn1.7_frame2idx.json', 'r') as f:
        frame2idx = json.load(f)
    y = []
    for instance in data:
        frame = False
        for i in instance[2]:
            if i != '_':
                frame = i
                break
        frameidx = frame2idx[frame]
        y.append(frameidx)
    return tuple(y)

def get_target_frames(input_data):
    all_y = dict(Counter(get_y(input_data)))
    target_frames = []
    for i in all_y:
        count = all_y[i]
        if count >= 5:
            target_frames.append(i)
            
    return target_frames
            
trn_target_frames = get_target_frames(trn)
dev_target_frames = get_target_frames(dev)
tst_target_frames = get_target_frames(tst)

print('trn_target_frames:', len(trn_target_frames))
print('dev_target_frames:', len(dev_target_frames))
print('tst_target_frames:', len(tst_target_frames))


# In[9]:


trn_batch_sampler = prototypical_batch_sampler.PrototypicalBatchSampler(classes_per_it=60, 
                                                                        num_support=5,
                                                                        target_frames=trn_target_frames, 
                                                                        def_data=def_data, def_y=def_y)

# trn_batch_sampler = prototypical_batch_sampler.PrototypicalBatchSampler(classes_per_it=4, 
#                                                                         num_support=2,
#                                                                         target_frames=trn_target_frames, 
#                                                                         def_data=def_data, def_y=def_y)


# In[10]:


dev_batch_sampler = prototypical_batch_sampler.PrototypicalBatchSampler(classes_per_it=5, 
                                                                        num_support=5, 
                                                                        target_frames=dev_target_frames, 
                                                                        def_data=def_data, def_y=def_y)

# dev_batch_sampler = prototypical_batch_sampler.PrototypicalBatchSampler(classes_per_it=4, 
#                                                                         num_support=2, 
#                                                                         target_frames=dev_target_frames, 
#                                                                         def_data=def_data, def_y=def_y)


# In[11]:


tst_batch_sampler = prototypical_batch_sampler.PrototypicalBatchSampler(classes_per_it=5, 
                                                                        num_support=5, 
                                                                        target_frames=tst_target_frames, 
                                                                        def_data=def_data, def_y=def_y)

# tst_batch_sampler = prototypical_batch_sampler.PrototypicalBatchSampler(classes_per_it=4, 
#                                                                         num_support=2, 
#                                                                         target_frames=tst_target_frames, 
#                                                                         def_data=def_data, def_y=def_y)


# In[12]:


def get_embs_from_episode(episode):
    support_embs = []
    query_embs = []
    
    support_y, query_y = [],[]
    
    for class_indice in episode:
        support_examples, query_examples = class_indice

        query_inputs, _, query_token_type_ids, query_masks = query_examples[0][0]
        query_inputs = query_inputs.view(1,len(query_inputs)).to(device)
        query_token_type_ids = query_token_type_ids.view(1,len(query_token_type_ids)).to(device)
        query_masks = query_masks.view(1,len(query_masks)).to(device)
        
        query_frame = query_examples[0][1]
        query_y.append(query_frame)
    
        with torch.no_grad():
            _, query_emb = frameBERT(query_inputs, 
                                  token_type_ids=query_token_type_ids, 
                                  attention_mask=query_masks)
            query_emb = query_emb.view(-1)
        query_embs.append(query_emb)
        
        support_inputs, support_token_type_ids, support_masks = [],[],[]
        for i in range(len(support_examples)):
            support_input, _, _, _, _, support_token_type_id, support_mask = support_examples[i][0]
            support_inputs.append(support_input)
            support_token_type_ids.append(support_token_type_id)
            support_masks.append(support_mask)
            
            support_frame = support_examples[i][1]
            support_y.append(support_frame)
            
        support_inputs = torch.stack(support_inputs).to(device)
        support_token_type_ids = torch.stack(support_token_type_ids).to(device)
        support_masks = torch.stack(support_masks).to(device)
        
        with torch.no_grad():
            _, support_emb = frameBERT(support_inputs, 
                                  token_type_ids=support_token_type_ids, 
                                  attention_mask=support_masks)
        support_embs.append(support_emb)
        
    support_embs = torch.stack(support_embs)
    support_embs = support_embs.view(-1, 768)
    query_embs = torch.stack(query_embs)
    
    support_y = tuple(support_y)
    query_y = tuple(query_y)
    
    return support_embs, query_embs, support_y, query_y


# In[13]:


def train(trn_batch, dev_batch,  best_model_path, last_model_path, model=False):
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
    
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    
    for epoch in range(TRN_EPOCHS):
        model.train()
        for episode in trn_batch:
            support_embs, query_embs, support_y, query_y = get_embs_from_episode(episode)
            
            support_embs = model(support_embs)
            query_embs = model(query_embs)

            loss, acc = loss_fn(support_embs, query_embs, support_y, query_y, len(support_y))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            model.zero_grad()
            
            train_loss.append(loss.item())
            train_acc.append(acc.item())

            
        avg_loss = np.mean(train_loss[-100:])
        avg_acc = np.mean(train_acc[-100:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        
        model.eval()
        for episode in dev_batch:
            support_embs, query_embs, support_y, query_y = get_embs_from_episode(episode)
            support_embs = model(support_embs)
            query_embs = model(query_embs)
            
            loss_val, acc_val = loss_fn(support_embs, query_embs, support_y, query_y, len(support_y))
            
            val_loss.append(loss_val.item())
            val_acc.append(acc_val.item())

            
        avg_loss = np.mean(val_loss[-100:])
        avg_acc = np.mean(val_acc[-100:])
        
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join('/disk/data/models/framenet/prototype_mlp/',
                                       name + '.txt'), locals()[name])
        
        
        
best_model_path = '/disk/data/models/framenet/prototype_mlp/best_model.pth'
last_model_path = '/disk/data/models/framenet/prototype_mlp/last_model.pth'
trn_y = get_y(trn)
dev_y = get_y(dev)

trn_batch = trn_batch_sampler.gen_batch(trn_data, trn_y)
dev_batch = dev_batch_sampler.gen_batch(dev_data, dev_y)

TRN_EPOCHS = 100
mlp_model = MLP()
mlp_model.to(device)

print('\n...training')
train(trn_batch, dev_batch, best_model_path, last_model_path, model=mlp_model)
print(tac())


# In[ ]:


def test(tst_batch, model_path=False):
    avg_acc = list()
    model = MLP()
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    for epoch in range(10):
        model.eval()
        for episode in tst_batch:
            support_embs, query_embs, support_y, query_y = get_embs_from_episode(episode)

            support_embs = model(support_embs)
            query_embs = model(query_embs)

            _, acc_val = loss_fn(support_embs, query_embs, support_y, query_y, len(support_y))

            avg_acc.append(acc_val.item())
        
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))
    with open('/disk/data/models/framenet/prototype_mlp/test_acc.txt','w') as f:
        f.write(str(avg_acc))
        
best_model_path = '/disk/data/models/framenet/prototype_mlp/best_model.pth'
tst_y = get_y(tst)
tst_batch = tst_batch_sampler.gen_batch(tst_data, tst_y)


print('\n...testing')
test(tst_batch, model_path=best_model_path)
print(tac())

