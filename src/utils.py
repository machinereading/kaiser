import json
import sys
import torch
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from torch import nn

sys.path.insert(0,'../')
sys.path.insert(0,'../../')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

MAX_LEN = 256

import os
try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'
    
dir_path = dir_path+'/..'

class for_BERT():
    def __init__(self, srl='framenet', language='ko', fnversion=1.1, mode='train', masking=True, pretrained='bert-base-multilingual-cased'):
        self.mode = mode
        self.masking = masking
        self.srl = srl
        
        # add <tgt> and </tgt> to special token
#         never_split = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", '<tgt>', '</tgt>']
        if 'multilingual' in pretrained:
            vocab_file_path = dir_path+'/data/bert-multilingual-cased-dict-add-tgt'
            self.tokenizer = BertTokenizer(vocab_file_path, do_lower_case=False, max_len=256)
    #         self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False, max_len=256)
            self.tokenizer.additional_special_tokens = ['<tgt>', '</tgt>']
        elif 'large' in pretrained:
            vocab_file_path = dir_path+'/data/bert-large-cased-dict-add-tgt'
            self.tokenizer = BertTokenizer(vocab_file_path, do_lower_case=False, max_len=256)
            self.tokenizer.additional_special_tokens = ['<tgt>', '</tgt>']
        else:
            vocab_file_path = dir_path+'/data/bert-multilingual-cased-dict-add-tgt'
            self.tokenizer = BertTokenizer(vocab_file_path, do_lower_case=False, max_len=256)
            self.tokenizer.additional_special_tokens = ['<tgt>', '</tgt>']
        
        if srl == 'framenet':
            if language == 'en':
                fnversion=1.7
                data_path = dir_path+'/koreanframenet/resource/info/fn'+str(fnversion)+'_'
            else:
                data_path = dir_path+'/koreanframenet/resource/info/kfn'+str(fnversion)+'_'
            with open(data_path+'lu2idx.json','r') as f:
                self.lu2idx = json.load(f)
            self.idx2lu = dict(zip(self.lu2idx.values(),self.lu2idx.keys()))
                
            fname = dir_path+'/koreanframenet/resource/info/fn1.7_frame2idx.json'
            with open(fname,'r') as f:
                self.sense2idx = json.load(f)
                
            with open(data_path+'lufrmap.json','r') as f:
                self.lufrmap = json.load(f)
                
            with open(dir_path+'/koreanframenet/resource/info/fn1.7_fe2idx.json','r') as f:
                self.arg2idx = json.load(f)
                
            with open(dir_path+'/koreanframenet/resource/info/fn1.7_bio_fe2idx.json','r') as f:
                self.bio_arg2idx = json.load(f)
            self.idx2bio_arg = dict(zip(self.bio_arg2idx.values(),self.bio_arg2idx.keys()))
                
        else:
            if language =='en':
                pred2idx_fname = dir_path+'/data/enpb-pred2idx.json'
                arg2idx_fname = dir_path+'/data/enpb-arg2idx.json'
            else:
                pred2idx_fname = dir_path+'/data/kopb-pred2idx.json'
                arg2idx_fname = dir_path+'/data/kopb-arg2idx.json'
            with open(dir_path+'/data/pb-pred2idx.json','r') as f:
                self.lu2idx = json.load(f)
            with open(pred2idx_fname, 'r') as f:
                self.sense2idx = json.load(f)
            with open(arg2idx_fname, 'r') as f:
                self.arg2idx = json.load(f)
            self.lufrmap = None
        with open(dir_path+'/koreanframenet/resource/info/fn1.7_frargmap.json','r') as f:
            self.frargmap = json.load(f)
        with open(dir_path+'/koreanframenet/resource/info/fn1.7_bio_fe2idx.json','r') as f:
            self.bio_arg2idx = json.load(f)
        with open(dir_path+'/koreanframenet/resource/info/fn1.7_bio_frargmap.json','r') as f:
            self.bio_frargmap = json.load(f)
            
        self.idx2sense = dict(zip(self.sense2idx.values(),self.sense2idx.keys()))
        self.idx2arg = dict(zip(self.arg2idx.values(),self.arg2idx.keys()))
#         self.idx2bio_arg = dict(zip(self.bio_arg2idx.values(),self.bio_arg2idx.keys()))
        
    def idx2tag(self, predictions, model='senseid'):
        if model == 'senseid':
            pred_tags = [self.idx2sense[p_i] for p in predictions for p_i in p]
        elif model == 'argid-dp':
            pred_tags = [self.idx2arg[p_i] for p in predictions for p_i in p]
        elif model == 'argid-span':
            pred_tags = [self.idx2bio_arg[p_i] for p in predictions for p_i in p]
        return pred_tags
    
    
    
    def bert_tokenizer(self, text):
        orig_tokens = text.split(' ')
        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append("[CLS]")
        for orig_token in orig_tokens:
            orig_to_tok_map.append(len(bert_tokens))
            bert_tokens.extend(self.tokenizer.tokenize(orig_token))
        bert_tokens.append("[SEP]")

        return orig_tokens, bert_tokens, orig_to_tok_map
    
    def convert_to_bert_input_JointShallowSemanticParsing(self, input_data):
        tokenized_texts, lus, senses, args = [],[],[],[]
        orig_tok_to_maps = []
        for i in range(len(input_data)):    
            data = input_data[i]
            text = ' '.join(data[0])
            orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer(text)
            orig_tok_to_maps.append(orig_to_tok_map)
            tokenized_texts.append(bert_tokens)

            ori_lus = data[1]    
            lu_sequence = []
            for i in range(len(bert_tokens)):
                if i in orig_to_tok_map:
                    idx = orig_to_tok_map.index(i)
                    l = ori_lus[idx]
                    lu_sequence.append(l)
                else:
                    lu_sequence.append('_')
            lus.append(lu_sequence)        

            if self.mode == 'train':
                ori_senses, ori_args = data[2], data[3]
                sense_sequence, arg_sequence = [],[]
                for i in range(len(bert_tokens)):
                    if i in orig_to_tok_map:
                        idx = orig_to_tok_map.index(i)
                        fr = ori_senses[idx]
                        sense_sequence.append(fr)
                        ar = ori_args[idx]
                        arg_sequence.append(ar)
                    else:
                        sense_sequence.append('_')
                        arg_sequence.append('X')
                senses.append(sense_sequence)
                args.append(arg_sequence)

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        orig_tok_to_maps = pad_sequences(orig_tok_to_maps, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post", value=-1)
        
        if self.mode =='train':
            if self.srl == 'propbank-dp':
                arg_ids = pad_sequences([[self.arg2idx.get(ar) for ar in arg] for arg in args],
                                        maxlen=MAX_LEN, value=self.arg2idx["X"], padding="post",
                                        dtype="long", truncating="post")                
            else:
                arg_ids = pad_sequences([[self.bio_arg2idx.get(ar) for ar in arg] for arg in args],
                                        maxlen=MAX_LEN, value=self.bio_arg2idx["X"], padding="post",
                                        dtype="long", truncating="post")

        lu_seq, sense_seq = [],[]
        for sent_idx in range(len(lus)):
            lu_items = lus[sent_idx]
            lu = []
            for idx in range(len(lu_items)):
                if lu_items[idx] != '_':
                    if len(lu) == 0:
                        if self.mode != 'train' and self.masking == False:
                            lu.append(1)
                        else:
                            lu.append(self.lu2idx[lu_items[idx]])
                            
            lu_seq.append(lu)
            
            if self.mode == 'train':
                sense_items, arg_items = senses[sent_idx], args[sent_idx]
                sense = []
                for idx in range(len(sense_items)):
                    if sense_items[idx] != '_':
                        if len(sense) == 0:
                            sense.append(self.sense2idx[sense_items[idx]])
                sense_seq.append(sense)

        attention_masks = [[float(i>0) for i in ii] for ii in input_ids]    
        data_inputs = torch.tensor(input_ids)
        data_orig_tok_to_maps = torch.tensor(orig_tok_to_maps)
        data_lus = torch.tensor(lu_seq)
        data_masks = torch.tensor(attention_masks)
        
        if self.mode == 'train':
            data_senses = torch.tensor(sense_seq)
            data_args = torch.tensor(arg_ids)
            bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_lus, data_senses, data_args, data_masks)
        else:
            bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_lus, data_masks)
        return bert_inputs

    
def get_masks(datas, mapdata, num_label=2, masking=True):
    masks = []
    if masking == True:
        for idx in datas:
            mask = torch.zeros(num_label)
    #         mask[mask==0] = np.NINF
            try:
                candis = mapdata[str(int(idx[0]))]
            except KeyboardInterrupt:
                raise
            except:
                candis = mapdata[int(idx[0])]
            for candi_idx in candis:
                mask[candi_idx] = 1
            masks.append(mask)
    else:
        for idx in datas:
            mask = torch.ones(num_label)
            masks.append(mask)
    masks = torch.stack(masks)
    return masks

def masking_logit(logit, mask):
    with torch.no_grad():
        masking = np.multiply(logit, mask)
    masking[masking==0] = np.NINF
    
    return masking
    

def logit2label(masked_logit):
    sm = nn.Softmax()
    pred_logits = sm(masked_logit).view(1,-1)
    score, label = pred_logits.max(1)
    score = float(score)
    
    return label, score

def get_tgt_idx(bert_tokens, tgt=False):
    tgt_idx = []
    try:
        if tgt == False:
            for i in range(len(bert_tokens)):
                if bert_tokens[i] == '<':
                    if bert_tokens[i+1] == 't' and bert_tokens[i+2] == '##gt' and bert_tokens[i+3] == '>':
                        tgt_idx.append(i)
                        tgt_idx.append(i+1)
                        tgt_idx.append(i+2)
                        tgt_idx.append(i+3)
                    elif bert_tokens[i+1] == '/' and bert_tokens[i+2] == 't' and bert_tokens[i+3] == '##gt' and bert_tokens[i+4] == '>':
                        tgt_idx.append(i)
                        tgt_idx.append(i+1)
                        tgt_idx.append(i+2)
                        tgt_idx.append(i+3)
                        tgt_idx.append(i+4)
        else:
            tgt_token_list = ['<tgt>', '</tgt>']
            for i in range(len(bert_tokens)):
                if bert_tokens[i] in tgt_token_list:
                    tgt_idx.append(i)
    except KeyboardInterrupt:
        raise
    except:
        pass
    
    return tgt_idx