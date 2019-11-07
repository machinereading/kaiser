
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
from seqeval.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


# In[2]:


srl = 'framenet'
# srl = 'propbank-dp'
language = 'ko'
language = 'en'
masking = True
model_dir = '/disk/data/models/enframenet_1105/'
if language == 'en':
    fnversion = 1.7
#     PRETRAINED_MODEL = "bert-large-cased"
    MAX_LEN = 256
    batch_size = 6
    PRETRAINED_MODEL = "bert-base-multilingual-cased"
else:
    fnversion = 1.1
    PRETRAINED_MODEL = "bert-base-multilingual-cased"
    MAX_LEN = 256
    batch_size = 6

epochs = 50


# In[14]:


try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'


# In[3]:


trn, dev, tst = dataio.load_data(srl=srl, language=language)
print('')
print('MODEL:', srl)
print('LANGUAGE:', language)


# In[16]:


bert_io = utils.for_BERT(mode='train', srl=srl, language=language, masking=True, fnversion=fnversion)


# In[9]:


def train():
    print('your model would be saved at', model_dir)
    
    if srl == 'propbank-dp':
        model = BertForJointShallowSemanticParsing.from_pretrained(PRETRAINED_MODEL, 
                                                         num_senses = len(bert_io.sense2idx), 
                                                         num_args = len(bert_io.arg2idx))
    else:
        model = BertForJointShallowSemanticParsing.from_pretrained(PRETRAINED_MODEL, 
                                                         num_senses = len(bert_io.sense2idx), 
                                                         num_args = len(bert_io.bio_arg2idx),
                                                         lufrmap=bert_io.lufrmap, frargmap = bert_io.bio_frargmap)
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
#         break

        # print train loss per epoch
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        model_saved_path = model_dir+'epoch-'+str(num_of_epoch)+'-joint.pt'        
        torch.save(model, model_saved_path)
        num_of_epoch += 1
    print('...training is done')


# In[ ]:


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def test():
    result_path = '/disk/data/models/'+model_dir.split('/')[-2]+'-result/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print('TEST result would be saved at:', result_path)
    models = glob.glob(model_dir+'*.pt')
    results = []
    for m in models:
        print('model:', m)
        model = torch.load(m)
        model.eval()

        tst_data = bert_io.convert_to_bert_input_JointShallowSemanticParsing(tst)
        sampler = RandomSampler(tst)
        tst_dataloader = DataLoader(tst_data, sampler=sampler, batch_size=batch_size)

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        
        pred_senses, true_senses, pred_args, true_args = [],[],[],[]
        for batch in tst_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_orig_tok_to_maps, b_lus, b_senses, b_args, b_masks = batch

            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids, token_type_ids=None, 
                                     lus=b_lus, attention_mask=b_masks)
                sense_logits, arg_logits = model(b_input_ids, token_type_ids=None, 
                                lus=b_lus, attention_mask=b_masks)
            sense_logits = sense_logits.detach().cpu().numpy()
            arg_logits = arg_logits.detach().cpu().numpy()
            
            gold_sense_ids = b_senses.to('cpu').numpy()
            gold_arg_ids = b_args.to('cpu').numpy()
            input_ids = b_input_ids.to('cpu').numpy()
            lufr_masks = utils.get_masks(b_lus, bert_io.lufrmap, num_label=len(bert_io.sense2idx)).to(device)
            
            for b_idx in range(len(sense_logits)):
                input_id = input_ids[b_idx]
                sense_logit = sense_logits[b_idx]
                arg_logit = arg_logits[b_idx]
                lufr_mask = lufr_masks[b_idx]
                orig_tok_to_map = b_orig_tok_to_maps[b_idx]
                
                pred_sense, sense_score = utils.logit2label(sense_logit, lufr_mask)
                frarg_mask = utils.get_masks([pred_sense], bert_io.bio_frargmap, num_label=len(bert_io.bio_arg2idx)).to(device)[0]

                pred_arg_bert = []
                for logit in arg_logit:
                    label, score = utils.logit2label(logit, frarg_mask)
                    pred_arg_bert.append(int(label))
                 
                #infer
                pred_arg,true_arg = [],[]
                for idx in orig_tok_to_map:
                    if idx != -1:
                        tok_id = int(input_id[idx])
                        if tok_id == 1:
                            pass
                        elif tok_id == 2:
                            pass
                        else:
                            pred_arg.append(pred_arg_bert[idx])
                            true_arg.append(gold_arg_ids[b_idx][idx])
                
                pred_senses.append([int(pred_sense)])
                pred_args.append(pred_arg)
                true_args.append(true_arg)
            true_senses.append(gold_sense_ids)
            
#             break
#         break

        pred_sense_tags = [bert_io.idx2sense[p_i] for p in pred_senses for p_i in p]
        valid_sense_tags = [bert_io.idx2sense[l_ii] for l in true_senses for l_i in l for l_ii in l_i]
        
        pred_arg_tags = [[bert_io.idx2bio_arg[p_i] for p_i in p] for p in pred_args]
        valid_arg_tags = [[bert_io.idx2bio_arg[v_i] for v_i in v] for v in true_args]

        acc = accuracy_score(pred_sense_tags, valid_sense_tags)
        f1 = f1_score(pred_arg_tags, valid_arg_tags)
        print("SenseId Accuracy: {}".format(accuracy_score(pred_sense_tags, valid_sense_tags)))
        print("ArgId F1: {}".format(f1_score(pred_arg_tags, valid_arg_tags)))
        
        result = m+'\tsenseid:'+str(acc)+'\targid:'+str(f1)+'\n'
        results.append(result)
        
        epoch = m.split('-')[1]
        fname = result_path+str(epoch)+'-result.txt'
        with open(fname, 'w') as f:
            line = result
            f.write(line)
            line = 'gold'+'\t'+'pred'+'\n'
            f.write(line)
            for r in range(len(pred_sense_tags)):
                line = valid_sense_tags[r] + '\t' + pred_sense_tags[r]+'\n'
                f.write(line)
                line = str(valid_arg_tags[r]) + '\t' + str(pred_arg_tags[r])+'\n'
                f.write(line)
    fname = result_path+'result.txt'
    with open(fname, 'w') as f:
        for r in results:
            f.write(r)

    print('result is written to', fname)


# In[11]:


# train()


# In[ ]:


# test()


# In[24]:


import tensorflow as tf
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = BertForSequenceClassification.from_pretrained('bert-large-cased')

