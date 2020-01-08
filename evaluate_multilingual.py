
# coding: utf-8

# In[1]:


import json
import os
import parser
from src import dataio
import glob
from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score, precision_score, recall_score
import random

import torch
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
if device != "cpu":
    torch.cuda.set_device(0)


# In[2]:


try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'


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


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

with open('./data/frame_coreFE_list.json','r') as f:
    frame_coreFE = json.load(f)

def weighting(frame, args):
    weighted_args = []
    for arg in args:
        weighted_args.append(arg)
        if arg in frame_coreFE[frame]:
            weighted_args.append(arg)
        else:
            pass
    return weighted_args


# In[5]:


def test(srl=False, masking=False, viterbi=False, language=False, model_path=False, 
         result_dir=False, train_lang=False, tgt=False,
         pretrained="bert-base-multilingual-cased"):
    if not result_dir:
        result_dir = '/disk/data/models/'+model_dir.split('/')[-2]+'-result/'
    else:
        pass
    if result_dir[-1] != '/':
        result_dir = result_dir+'/'
        
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    if not train_lang:
        train_lang = language
    
    fname = fname = result_dir+train_lang+'_for_'+language
        
    if masking:
        fname = fname + '_with_masking_result.txt'
    else:
        fname = fname +'_result.txt'
        
    print('### Your result would be saved to:', fname)
        
    trn, dev, tst = dataio.load_data(srl=srl, language=language, exem=False)
    
    print('### EVALUATION')
    print('MODE:', srl)
    print('target LANGUAGE:', language)
    print('trained LANGUAGE:', train_lang)
    print('Viterbi:', viterbi)
    print('masking:', masking)
    print('using TGT token:', tgt)
    tic()    
        
    models = glob.glob(model_path+'*/')
    
#     models = []
    
    # en_exemplar best
#     models.append('/disk/data/models/dict_framenet/enModel-with-exemplar/9/')
#     models.append('/disk/data/models/frameBERT/frameBERT_en/')
    
#     # ko best
#     models.append('/disk/data/models/framenet/koModel/35/')
    
    # mul best
#     models.append('/disk/data/models/framenet_old/mulModel-100/39/')
#     models.append('/disk/data/models/dict_framenet/mulModel-100/39/')

#     mul best
#     models.append('/disk/data/models/framenet_old/mulModel-100/39/')
#     models.append(model_path+'33/')
    
    eval_result = []
    for m in models:
#         m = '/disk/data/models/framenet/enModel-with-exemplar/epoch-8-joint.pt'
        print('### model dir:', m)
        print('### TARGET LANGUAGE:', language)
        torch.cuda.set_device(device)
        model = parser.ShallowSemanticParser(srl=srl,gold_pred=True, model_path=m, viterbi=viterbi, 
                                             masking=masking, language='multilingual', tgt=tgt,
                                             pretrained=pretrained)

        gold_senses, pred_senses, gold_args, pred_args = [],[],[],[]        
        gold_full_all, pred_full_all = [],[]

        for instance in tst:
            torch.cuda.set_device(device)
#             try:
            result = model.parser(instance)

            gold_sense = [i for i in instance[2] if i != '_'][0]
            pred_sense = [i for i in result[0][2] if i != '_'][0]


            gold_arg = [i for i in instance[3] if i != 'X']
            pred_arg = [i for i in result[0][3]]

            gold_senses.append(gold_sense)
            pred_senses.append(pred_sense)
            
            gold_args.append(gold_arg)
            pred_args.append(pred_arg)

            if srl == 'framenet':
                gold_full = []
                gold_full.append(gold_sense)
                gold_full.append(gold_sense)
                weighted_gold_args = weighting(gold_sense, gold_arg)
                gold_full += weighted_gold_args

                pred_full = []
                pred_full.append(pred_sense)
                pred_full.append(pred_sense)
                weighted_pred_args = weighting(pred_sense, pred_arg)
                pred_full += weighted_pred_args

                gold_full_all.append(gold_full)
                pred_full_all.append(pred_full)
                


                
                    
#             except KeyboardInterrupt:
#                 raise
#             except:
#                 print("cuda error")
#                 pass
            
#             break
            
        acc = accuracy_score(gold_senses, pred_senses)
        arg_f1 = f1_score(gold_args, pred_args)
        arg_precision = precision_score(gold_args, pred_args)
        arg_recall = recall_score(gold_args, pred_args)
        

#         epoch = m.split('/')[-1].split('-')[1]
        epoch = m.split('/')[-2]
        print('# EPOCH:', epoch)
        print("SenseId Accuracy: {}".format(acc))
        print("ArgId Precision: {}".format(arg_precision))
        print("ArgId Recall: {}".format(arg_recall))
        print("ArgId F1: {}".format(arg_f1))
        if srl == 'framenet':
            full_f1 = f1_score(gold_full_all, pred_full_all)
            full_precision = precision_score(gold_full_all, pred_full_all)
            full_recall = recall_score(gold_full_all, pred_full_all)
            print("full-structure Precision: {}".format(full_precision))
            print("full-structure Recall: {}".format(full_recall))
            print("full-structure F1: {}".format(full_f1))
        print('-----processing time:', tac())
        print('')


        model_result = []
        model_result.append(epoch)
        model_result.append(acc)
        model_result.append(arg_precision)
        model_result.append(arg_recall)
        model_result.append(arg_f1)
        if srl == 'framenet':
            model_result.append(full_precision)
            model_result.append(full_recall)
            model_result.append(full_f1)
        model_result = [str(i) for i in model_result]
        eval_result.append(model_result)
    
    
    with open(fname,'w') as f:
        if srl == 'framenet':
            f.write('epoch'+'\t''SenseID'+'\t'+'Arg_P'+'\t'+'Arg_R'+'\t'+'ArgF1'+'\t'+'full_P'+'\t'+'full_R'+'\t'+'full_F1'+'\n')
        else:
            f.write('epoch'+'\t''SenseID'+'\t'+'Arg_P'+'\t'+'Arg_R'+'\t'+'ArgF1'+'\n')
        for i in eval_result:
            line = '\t'.join(i)
            f.write(line+'\n')
            
        print('\n\t### Your result is saved at:', fname)


# # eval for 25%

# In[6]:


srl = 'framenet'
language = 'ko'


# In[7]:


# print('\t###eval for ko Model (masking)')
# model_path = '/disk/data/models/framenet/koModel-25/'
# result_dir = '/disk/data/models/eval_result-25'

# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='ko', 
#      model_path=model_path, result_dir=result_dir)


# In[7]:


print('\t###eval for proto-distill (masking)')
model_path = '/disk/data/models/framenet/proto_distilling-25/'
result_dir = '/disk/data/models/eval_result-25'
test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='proto_distilling', 
     model_path=model_path, result_dir=result_dir)


# In[8]:


# print('\t###multilingual-for-ko (masking)')
# model_path = '/disk/data/models/framenet/mulModel-25/'
# result_dir = '/disk/data/models/eval_result-25'
# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='mul', 
#      model_path=model_path, result_dir=result_dir)


# In[10]:


# print('\t###eval for ko Model (no masking)')
# model_path = '/disk/data/models/framenet/koModel-25/'
# result_dir = '/disk/data/models/eval_result-25'

# test(srl=srl, language=language, masking=False, viterbi=False, tgt=True, train_lang='ko', 
#      model_path=model_path, result_dir=result_dir)


# In[9]:


# print('\t###eval for proto-distill (no masking)')
# model_path = '/disk/data/models/framenet/proto_distilling-25/'
# result_dir = '/disk/data/models/eval_result-25'
# test(srl=srl, language=language, masking=False, viterbi=False, tgt=True, train_lang='proto_distilling', 
#      model_path=model_path, result_dir=result_dir)


# In[10]:


# print('\t###multilingual-for-ko (no masking)')
# model_path = '/disk/data/models/framenet/mulModel-25/'
# result_dir = '/disk/data/models/eval_result-25'
# test(srl=srl, language=language, masking=False, viterbi=False, tgt=True, train_lang='mul', 
#      model_path=model_path, result_dir=result_dir)


# # eval for 100 (ko)

# In[13]:


srl = 'framenet'


# In[14]:


# srl = 'framenet'
# language = 'ko'
# print('\t###eval for ko Model (masking)')
# model_path = '/disk/data/models/framenet/koModel/'
# result_dir = '/disk/data/models/eval_result-100'
# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='ko', 
#      model_path=model_path, result_dir=result_dir)


# In[15]:


# print('\t###eval for proto-distill (masking)')
# language = 'ko'
# model_path = '/disk/data/models/framenet/proto_distilling/'
# result_dir = '/disk/data/models/eval_result-100'
# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='proto_distilling', 
#      model_path=model_path, result_dir=result_dir)


# In[16]:


# print('\t###multilingual-for-ko (masking)')
# language = 'ko'
# model_path = '/disk/data/models/dict_framenet/mulModel-100/'
# result_dir = '/disk/data/models/eval_result-100'
# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='mul', 
#      model_path=model_path, result_dir=result_dir)


# In[17]:


# print('\t###eval for ko Model (no masking)')
# model_path = '/disk/data/models/framenet/koModel/'
# result_dir = '/disk/data/models/eval_result-100'
# test(srl=srl, language=language, masking=False, viterbi=False, tgt=True, train_lang='ko', 
#      model_path=model_path, result_dir=result_dir)


# In[18]:


# print('\t###eval for proto-distill (no masking)')
# model_path = '/disk/data/models/framenet/proto_distilling/'
# result_dir = '/disk/data/models/eval_result-100'
# test(srl=srl, language=language, masking=False, viterbi=False, tgt=True, train_lang='proto_distilling', 
#      model_path=model_path, result_dir=result_dir)


# In[19]:


# print('\t###multilingual-for-ko (no masking)')
# # model_path = '/disk/data/models/framenet/mulModel-100/'
# model_path = '/disk/data/models/framenet/mulModel-100/'
# result_dir = '/disk/data/models/eval_result-100'
# test(srl=srl, language=language, masking=False, viterbi=False, tgt=True, train_lang='mul', 
#      model_path=model_path, result_dir=result_dir)


# ## eval for 10%

# In[20]:


# print('\t###eval for ko Model (masking)')
# language = 'ko'
# model_path = '/disk/data/models/framenet/koModel-10/'
# result_dir = '/disk/data/models/eval_result-10'
# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='ko', 
#      model_path=model_path, result_dir=result_dir)


# In[21]:


print('\t###eval for proto-distill (masking)')
language = 'ko'
model_path = '/disk/data/models/framenet/proto_distilling-10/'
result_dir = '/disk/data/models/eval_result-10'
test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='proto_distilling', 
     model_path=model_path, result_dir=result_dir)


# In[22]:


# print('\t###multilingual-for-ko (masking)')
# language = 'ko'
# model_path = '/disk/data/models/framenet/mulModel-10/'
# result_dir = '/disk/data/models/eval_result-10'
# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='mul', 
#      model_path=model_path, result_dir=result_dir)


# In[23]:


# print('\t###eval for ko Model (no masking)')
# language = 'ko'
# model_path = '/disk/data/models/framenet/koModel-10/'
# result_dir = '/disk/data/models/eval_result-10'
# test(srl=srl, language=language, masking=False, viterbi=False, tgt=True, train_lang='ko', 
#      model_path=model_path, result_dir=result_dir)


# In[24]:


# print('\t###eval for proto-distill (no masking)')
# language = 'ko'
# model_path = '/disk/data/models/framenet/proto_distilling-10/'
# result_dir = '/disk/data/models/eval_result-10'
# test(srl=srl, language=language, masking=False, viterbi=False, tgt=True, train_lang='proto_distilling', 
#      model_path=model_path, result_dir=result_dir)


# In[25]:


# print('\t###multilingual-for-ko (no masking)')
# language = 'ko'
# model_path = '/disk/data/models/framenet/mulModel-10/'
# result_dir = '/disk/data/models/eval_result-10'
# test(srl=srl, language=language, masking=False, viterbi=False, tgt=True, train_lang='mul', 
#      model_path=model_path, result_dir=result_dir)


# # eval for proto-distill (ko)

# In[26]:


# print('\t###eval for proto-distill (masking)')
# srl = 'framenet'
# language = 'ko'
# model_path = '/disk/data/models/framenet/proto_distilling/'

# result_dir = '/disk/data/models/proto_distilling/'
# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='prot-_distilling', 
#      model_path=model_path, result_dir=result_dir)


# In[27]:


# print('\t###multilingual-for-ko-no-masking')
# srl = 'framenet'
# language = 'ko'
# model_path = '/disk/data/models/framenet/mulModel-100/'

# result_dir = '/disk/data/models/eval_result/'
# test(srl=srl, language=language, masking=False, viterbi=False, tgt=True, train_lang='mul', 
#      model_path=model_path, result_dir=result_dir)


# In[28]:


# print('\t###eval for proto-distill (masking)')
# srl = 'framenet'
# language = 'en'
# model_path = '/disk/data/models/framenet/proto_distilling/'

# result_dir = '/disk/data/models/proto_distilling/'
# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='prot-_distilling', 
#      model_path=model_path, result_dir=result_dir)


# # eval for en for en

# In[29]:


# print('\t###multilingual-for-en-masking')
# srl = 'framenet'
# language = 'en'
# model_path = '/disk/data/models/framenet/enModel-with-exemplar/'

# result_dir = '/disk/data/models/eval_result/'
# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='en_with_exem', 
#      model_path=model_path, result_dir=result_dir)


# # eval for ko for ko

# In[30]:


# print('\t###multilingual-for-en-masking')
# srl = 'framenet'
# language = 'en'
# model_path = '/disk/data/models/framenet/enModel-with-exemplar/'

# result_dir = '/disk/data/models/results/framenet/enModel-with-exemplar/'
# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='en_with_exem', 
#      model_path=model_path, result_dir=result_dir)


# In[31]:


# print('\t###multilingual-for-ko-masking')
# srl = 'framenet'
# language = 'ko'
# model_path = '/disk/data/models/framenet/koModel/'
# result_dir = '/disk/data/models/results/framenet/koModel/'
# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='ko', 
#      model_path=model_path, result_dir=result_dir)


# In[32]:


# print('\t###multilingual-for-en-without-masking')
# srl = 'framenet'
# language = 'en'
# model_path = '/disk/data/models/framenet/enModel-with-exemplar/'
# result_dir = '/disk/data/models/results/framenet/enModel-with-exemplar/'
# test(srl=srl, language=language, masking=False, viterbi=False, tgt=True, train_lang='en_with_exem', 
#      model_path=model_path, result_dir=result_dir)


# In[33]:


# print('\t###ko-for-ko-without-masking')
# srl = 'framenet'
# language = 'ko'
# model_path = '/disk/data/models/framenet/koModel/'
# result_dir = '/disk/data/models/results/framenet/koModel/'
# test(srl=srl, language=language, masking=False, viterbi=False, tgt=True, train_lang='ko', 
#      model_path=model_path, result_dir=result_dir)


# In[34]:


# print('\t###en-for-ko-masking')
# srl = 'framenet'
# language = 'ko'
# model_path = ''
# result_dir = '/disk/data/models/results/framenet/enModel-with-exemplar/'
# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='en_with_exem', 
#      model_path=model_path, result_dir=result_dir)


# In[35]:


# print('\t###en-for-ko-without-masking')
# srl = 'framenet'
# language = 'ko'
# model_path = ''
# result_dir = '/disk/data/models/results/framenet/enModel-with-exemplar/'
# test(srl=srl, language=language, masking=False, viterbi=False, tgt=True, train_lang='en_with_exem', 
#      model_path=model_path, result_dir=result_dir)


# # eval for KFN

# In[36]:


# print('\t###multilingual-for-ko-masking')
# srl = 'framenet'
# language = 'ko'
# model_path = '/disk/data/models/framenet/mulModel-100/'

# result_dir = '/disk/data/models/eval_result/'
# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='mul', 
#      model_path=model_path, result_dir=result_dir)


# In[37]:


# print('\t###multilingual-for-ko-without-masking')
# srl = 'framenet'
# language = 'ko'
# model_path = '/disk/data/models/framenet/mulModel-100/'
# model_path = '/disk/data/models/framenet/mulModel-100/'

# result_dir = '/disk/data/models/results/framenet/mulModel-100/'
# test(srl=srl, language=language, masking=False, viterbi=False, tgt=True, train_lang='en_ko', 
#      model_path=model_path, result_dir=result_dir)


# # eval for En again using mulModel

# In[38]:


# print('\t###multilingual-for-en-masking')
# srl = 'framenet'
# language = 'en'
# model_path = '/disk/data/models/framenet/mulModel-100/'

# result_dir = '/disk/data/models/eval_result/'
# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='mul', 
#      model_path=model_path, result_dir=result_dir)


# In[39]:


# print('\t###multilingual-for-en-without-masking')
# srl = 'framenet'
# language = 'en'
# model_path = '/disk/data/models/framenet/mulModel-100/'
# model_path = '/disk/data/models/framenet/mulModel-100/'

# result_dir = '/disk/data/models/results/framenet/mulModel-100-for-en/'
# test(srl=srl, language=language, masking=False, viterbi=False, tgt=True, train_lang='en_ko', 
#      model_path=model_path, result_dir=result_dir)


# # eval for distilling

# In[40]:


# print('\t###multilingual-for-ko-masking')
# srl = 'framenet'
# language = 'ko'
# model_path = '/disk/data/models/framenet/distilling/'

# result_dir = '/disk/data/models/distilling/'
# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='distilling', 
#      model_path=model_path, result_dir=result_dir)

