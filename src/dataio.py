import json
import sys
import glob
import torch
sys.path.insert(0,'../')
sys.path.insert(0,'../../')
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

from kaiser.koreanframenet import koreanframenet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

import os
try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'
    
dir_path = dir_path+'/..'
    
def conll2tagseq(data):
    tokens, preds, senses, args = [],[],[],[]
    result = []
    for line in data:
        line = line.strip()
        if line.startswith('#'):
            pass
        elif line != '':
            t = line.split('\t')
            token, pred, sense, arg = t[1], t[2], t[3], t[4]            
            tokens.append(token)
            preds.append(pred)
            senses.append(sense)
            args.append(arg)
        else:            
            sent = []
            sent.append(tokens)
            sent.append(preds)
            sent.append(senses)
            sent.append(args)            
            result.append(sent)
            tokens, preds, senses, args = [],[],[],[]
            
    return result
    
def load_data(srl='framenet', language='ko', fnversion=1.1, path=False):
    if srl == 'framenet':
        if language == 'ko':
            kfn = koreanframenet.interface(version=fnversion)
            trn_d, dev_d, tst_d = kfn.load_data()
        else:
            if path == False:
                fn_dir = '/disk/FrameNet/resource/fn1.7/'
            else:
                fn_dir = path
            with open(fn_dir+'fn1.7.fulltext.train.syntaxnet.conll') as f:
                d = f.readlines()
            trn_d = conll2tagseq(d)
            with open(fn_dir+'fn1.7.dev.syntaxnet.conll') as f:
                d = f.readlines()
            dev_d = conll2tagseq(d)
            with open(fn_dir+'fn1.7.test.syntaxnet.conll') as f:
                d = f.readlines()
            tst_d = conll2tagseq(d)
    else:
        if language == 'ko':
            if path == False:
                fn_dir = '/disk/data/corpus/koreanPropBank/revised/'
            else:
                fn_dir = path
            if srl == 'propbank-dp':
                
                with open(fn_dir+'srl.dp_based.train.conll') as f:
                    d = f.readlines()
                trn_d = conll2tagseq(d)
                with open(fn_dir+'srl.dp_based.test.conll') as f:
                    d = f.readlines()
                tst_d = conll2tagseq(d)
                dev_d = []
            else:
                with open(fn_dir+'srl.span_based.train.conll') as f:
                    d = f.readlines()
                trn_d = conll2tagseq(d)
                with open(fn_dir+'srl.span_based.test.conll') as f:
                    d = f.readlines()
                tst_d = conll2tagseq(d)
                dev_d = []
    trn = data2tgt_data(trn_d, mode='train')
    tst = data2tgt_data(tst_d, mode='train')
    if dev_d:
        dev = data2tgt_data(dev_d, mode='train')
    else:
        dev = []
        
    print('# of instances in trn:', len(trn))
    print('# of instances in dev:', len(dev))
    print('# of instances in tst:', len(tst))
    print('data example:', trn[0])
    
    return trn, dev, tst
                
def data2tgt_data(input_data, mode=False):
    result = []
    for item in input_data:
        
        if mode == 'train':
            ori_tokens, ori_preds, ori_senses, ori_args = item[0],item[1],item[2],item[3]
        else:
            ori_tokens, ori_preds = item[0],item[1]
        for idx in range(len(ori_preds)):
            pred = ori_preds[idx]
            if pred != '_':
                if idx == 0:
                    begin = idx
                elif ori_preds[idx-1] == '_':
                    begin = idx
                end = idx
                
        tokens, preds, senses, args = [],[],[],[]
        for idx in range(len(ori_preds)):
            token = ori_tokens[idx]
            pred = ori_preds[idx]
            if mode == 'train':
                sense = ori_senses[idx]
                arg = ori_args[idx]
                
            if idx == begin:
                tokens.append('<tgt>')
                preds.append('_')
                if mode == 'train':
                    senses.append('_')
                    args.append('X')

            tokens.append(token)
            preds.append(pred)
            
            if mode == 'train':
                senses.append(sense)
                args.append(arg)

            if idx == end:
                tokens.append('</tgt>')
                preds.append('_')
                
                if mode == 'train':
                    senses.append('_')
                    args.append('X')
        sent = []
        sent.append(tokens)
        sent.append(preds)
        if mode == 'train':
            sent.append(senses)
            sent.append(args)
            
        result.append(sent)
    return result

def preprocessor(input_data):
    if type(input_data) == str:
        data = input_data.split(' ')
        result = []
        result.append(data)
    elif type(input_data) == list:
        result = input_data
    else:
        data = input_data['text'].split(' ')
        result = []
        result.append(data)
    return result
                
            
            
            
    