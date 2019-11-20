
# coding: utf-8

# In[1]:


import os
import json
from collections import Counter
import jpype
import sys
sys.path.append('../')


# In[2]:


from kaiser.koreanframenet import koreanframenet
kfn = koreanframenet.interface(1.1)
from konlpy.tag import Kkma
kkma = Kkma()

try:
    target_dir = os.path.dirname( os.path.abspath( __file__ ))
except:
    target_dir = '.'


# In[27]:


class targetIdentifier():
    def __init__(self, srl='framenet', language='ko', only_lu=True):
        self.srl = srl
        self.language = language
        self.only_lu = only_lu
        
        with open(target_dir+'/data/targetdic-1.1.json','r') as f:
            targetdic = json.load(f)
        self.targetdic = targetdic
    
    def targetize(self, word):
        jpype.attachThreadToJVM()
        target_candis = []
        morps = kkma.pos(word)
        v = False
        for m,p in morps:
            if p == 'XSV' or p == 'VV' or p == 'VA':
                v = True    

        if v:
            for i in range(len(morps)):
                m,p = morps[i]
                if p == 'VA' or p == 'VV':
                    if p == 'VV':
                        pos = 'v'
                    elif p == 'VA':
                        pos = 'a'
                    else:
                        pos = 'v'
                        
                    if m[0] == word[0] and len(m) >= 1:
                        target_candis.append((m,pos))
#                 if p == 'NNG' or p == 'NNP':
                if p == 'NNG':
                    pos = 'n'
                    if m[0] == word[0] and len(m) >= 1:
                        target_candis.append((m,pos))
                if i > 0 and p == 'XSV':
                    pos = 'v'
                    if m[0] == word[0] and len(m) >= 1:
                        target_candis.append((m,pos))
                    r = morps[i-1][0]+m
                    if r[0] == word[0]:
                        target_candis.append((r,pos))
        else:
            pos = 'n'
            pos_list = []
            for m,p in morps:
                if p.startswith('J'):
                    pos_list.append(m)
                elif p == 'VCP' or p == 'EFN':
                    pos_list.append(m)
            for m, p in morps:
#                 if p == 'NNG' or p == 'NNP':
                if p == 'NNG':
                    if len(pos_list) == 0:
                        if m == word:
                            target_candis.append((m, pos))
                    else:
                        if m[0] == word[0]:
                            target_candis.append((m, pos))
        return target_candis

    def get_lu_by_token(self, token):
        target_candis = self.targetize(token)
        lu_candis = []
        for target_candi, word_pos in target_candis:
            for lu in self.targetdic:
                if target_candi in self.targetdic[lu]:
                    lu_pos = lu.split('.')[-1]
                    if word_pos == lu_pos:
                        lu_candis.append(lu)
            if self.only_lu==False:
                lu_candis.append(target_candi+'.'+word_pos)
        common = Counter(lu_candis).most_common()
        if len(common) > 0:
            result = common[0][0]
        else:
            result = False
        return result
    
    def target_id(self, input_conll):
        result = []
        tokens = input_conll[0]
        for idx in range(len(tokens)):
            token = tokens[idx]
            lu = self.get_lu_by_token(token)
            lus = ['_' for i in range(len(tokens))]
            if lu:
                lus[idx] = lu
                instance = []            
    #             instance.append(idxs)
                instance.append(tokens)
                instance.append(lus)
                result.append(instance)
        return result
    
    def pred_id(self, input_conll):
        result = []
        tokens = input_conll[0]
        for idx in range(len(tokens)):
            token = tokens[idx]
            lus = ['_' for i in range(len(tokens))]
            target_candis = self.targetize(token)
            for target_candi, word_pos in target_candis:
                if word_pos == 'v' or word_pos == 'a':
                    lus[idx] = 'PRED'
                    instance = []
                    instance.append(tokens)
                    instance.append(lus)
                    result.append(instance)
        return result


# In[35]:


# text = '애플은 스티브 잡스와 스티브 워즈니악과 론 웨인이 1976년에 설립한 컴퓨터 회사이다.'
# text = '헤밍웨이는 1899년 7월 21일 미국 일리노이에서 태어났고 62세에 자살로 사망했다.'
# text = '헤밍웨이는 풀린 파이퍼와 이혼한 뒤 마사 겔혼과 재혼하였다'
# text = '애플은 스티브 잡스와 스티브 워즈니악과 론 웨인이 1976년에 설립한 회사이다.'
# text = '헤밍웨이는 태어났고 마사 겔혼과 이혼하였다.'
# text = '헤밍웨이는 풀린 파이퍼와 이혼한 뒤 마사 겔혼과 재혼하였다'
# text = '멜 깁슨이 출연한 영화를 제작한 사람은 누구인가?'
# targetid = targetIdentifier(only_lu=False)

# i = [text.split(' ')]
# # d = targetid.pred_id(i)
# # print(d)
# # print('')
# d = targetid.target_id(i)
# from pprint import pprint
# pprint(d)

