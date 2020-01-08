import numpy as np
import torch
import os
import json
import sys
import random
from collections import Counter

sys.path.insert(0,'../')
sys.path.insert(0,'../../')

from kaiser.src import dataio
from kaiser.src import utils

try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'

class PrototypicalBatchSampler():
    
    def __init__(self, trn=False, tst=False, input_data=False, def_data=False, def_y=False,
                 classes_per_it=60, num_support=5, 
                 iterations=100, target_frames=False):
        with open(dir_path+'/../koreanframenet/resource/info/fn1.7_frame2idx.json', 'r') as f:
            self.frame2idx = json.load(f)
        with open(dir_path+'/../koreanframenet/resource/info/fn1.7_frame_definitions.json', 'r') as f:
            self.frame2definition = json.load(f)
            
        if target_frames:
            self.target_frames = target_frames
        else:
            with open(dir_path+'/../data/target_frames.json','r') as f:
                self.target_frames = json.load(f)

        self.idx2frame = dict(zip(self.frame2idx.values(),self.frame2idx.keys()))

#         self.trn = trn
#         self.tst = tst
        
#         self.trn_y = self.get_y(self.trn)
#         self.tst_y = self.get_y(self.tst)
        
#         self.trn_data = trn_data
#         self.tst_data = tst_data
        
        self.bert_io = utils.for_BERT(mode='train', language='multi')
        
        if def_data:
            self.def_data = def_data
            self.def_y = def_y
        else:
            self.def_data, self.def_y = self.bert_io.convert_to_bert_input_label_definition(self.frame2definition, self.frame2idx)
        
        self.classes_per_it = classes_per_it     
        self.num_support = num_support  
        self.iterations = iterations
        
        
    def frameidx2definition(self, frameidx):
        frameidx = int(frameidx)
        
        return self.frame2definition[self.idx2frame[frameidx]]
    
    def get_y(self, data):
        y = []
        for instance in data:
            frame = False
            for i in instance[2]:
                if i != '_':
                    frame = i
                    break
            frameidx = self.frame2idx[frame]
            y.append(frameidx)
        return tuple(y)
    
    def get_examples(self, data, y, frame, n_sample):
        all_examples = [] #example idx
        for idx in range(len(y)):
            if y[idx] == frame:
                all_examples.append(idx)
        example_idxs = random.sample(all_examples, k=n_sample)
        
        examples = []
        for idx in example_idxs:
            example = (data[idx], y[idx])
            examples.append(example)
        return examples
    
    def gen_episode(self, data, y, n_classes):
        classes = random.sample(self.target_frames, k=n_classes)
        
        episode = []
        for k in classes:
            frame = k
            support_examples = self.get_examples(data, y, frame, self.num_support)
            query_examples = self.get_examples(self.def_data, self.def_y, frame, 1)
            
            class_indice = (support_examples, query_examples)
            episode.append(class_indice)
        
        return episode
    
    def gen_batch(self, data, y):
        batch = []
        for i in range(self.iterations):
            episode = self.gen_episode(data, y, self.classes_per_it)
            batch.append(episode)
            
        return batch
            
            
                    
            
            
    

        
        
    
# class PrototypicalBatchSampler_ori(object):
#     '''
#     PrototypicalBatchSampler: yield a batch of indexes at each iteration.
#     Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
#     In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
#     for 'classes_per_it' random classes.
#     __len__ returns the number of episodes per epoch (same as 'self.iterations').
#     '''

#     def __init__(self, labels, classes_per_it, num_samples, iterations):
#         '''
#         Initialize the PrototypicalBatchSampler object
#         Args:
#         - labels: an iterable containing all the labels for the current dataset
#         samples indexes will be infered from this iterable.
#         - classes_per_it: number of random classes for each iteration
#         - num_samples: number of samples for each iteration for each class (support + query)
#         - iterations: number of iterations (episodes) per epoch
#         '''
#         super(PrototypicalBatchSampler, self).__init__()
#         self.labels = labels
#         self.classes_per_it = classes_per_it
#         self.sample_per_class = num_samples
#         self.iterations = iterations

#         self.classes, self.counts = np.unique(self.labels, return_counts=True)
#         self.classes = torch.LongTensor(self.classes)

#         # create a matrix, indexes, of dim: classes X max(elements per class)
#         # fill it with nans
#         # for every class c, fill the relative row with the indices samples belonging to c
#         # in numel_per_class we store the number of samples for each class/row
#         self.idxs = range(len(self.labels))
#         self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
#         self.indexes = torch.Tensor(self.indexes)
#         self.numel_per_class = torch.zeros_like(self.classes)
#         for idx, label in enumerate(self.labels):
#             label_idx = np.argwhere(self.classes == label).item()
#             self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
#             self.numel_per_class[label_idx] += 1

#     def __iter__(self):
#         '''
#         yield a batch of indexes
#         '''
#         spc = self.sample_per_class
#         cpi = self.classes_per_it

#         for it in range(self.iterations):
#             batch_size = spc * cpi
#             batch = torch.LongTensor(batch_size)
#             c_idxs = torch.randperm(len(self.classes))[:cpi]
#             for i, c in enumerate(self.classes[c_idxs]):
#                 s = slice(i * spc, (i + 1) * spc)
#                 # FIXME when torch.argwhere will exists
#                 label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
#                 sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
#                 batch[s] = self.indexes[label_idx][sample_idxs]
#             batch = batch[torch.randperm(len(batch))]
#             yield batch

#     def __len__(self):
#         '''
#         returns the number of iterations (episodes) per epoch
#         '''
#         return self.iterations