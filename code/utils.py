import tensorflow as tf
import numpy as np
import re
import random
import json
import glob
import codecs
import os
from tqdm import tqdm
from collections import defaultdict

np.random.seed(20190525)
random.seed(20190525)

def get_biaffine_predicate(pred_text,scores,label_list,predicate_labels,threshold=0):
    l = len(pred_text)
    size = l * (l+1) // 2
    def get_position(n,k):
        def prefix_sum(i):
            return (n + n - i) * (i + 1) // 2
        
        left,right=0,n
        while left < right:
            mid = (left + right) // 2
            if prefix_sum(mid) < k:
                left = mid + 1
            else:
                right = mid
            
        s = left
        e = k - s * n + s * (s + 1) // 2
        return (s,e)
    
    tags = []
    entities = defaultdict(set)
    for pos, lpos in np.argwhere(scores > threshold):
        lb = label_list[lpos]
        s,e = get_position(l,pos)
        if 'EH2ET' in lb:
            if s <= e:
                entities[s].add((s,e,lb[:-5]))
        else:
            tags.append((s,e,lb))

    results = []
    for p in predicate_labels:
        Hs = []
        Ho = []
        T = set()
        for s,e,t in tags:
            tp = t[-5:]
            tt = t[:-5]
            if tt != p:
                continue

            if tp == 'SH2OH':
                Hs.extend(entities.get(s,[]))
                Ho.extend(entities.get(e,[]))
            if tp == 'OH2SH':
                Ho.extend(entities.get(s,[]))
                Hs.extend(entities.get(e,[]))
            if tp == 'ST2OT':
                T.add((s,e))
            if tp == 'OT2ST':
                T.add((e,s))
        
        for s in set(Hs):
            for o in set(Ho):
                if (s[1],o[1]) in T:
                    results.append((s,p,o))

    return results
