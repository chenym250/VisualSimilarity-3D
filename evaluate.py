# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:12:39 2016

@author: chenym
"""

from load_save_objects import load_obj
import numpy as np

result = load_obj('results\\'+ 'query_result_train_size_907_test_size_200_1113_1213')

test_cat = load_obj('test_cla_shapedictionary')

test_cat_tree = load_obj('test_cla_shapedictionary')

train_cat = load_obj('train_cla_shapedictionary')

train_dist = load_obj('train_cla_shapedistribution')

test_dist = load_obj('test_cla_shapedistribution')
test_dist_low = load_obj('test_cla_shapedistribution_lowlevel')

#for k in result.keys():
#    result[k] = result[k][0:5]

def checkmatch(matchorder): # uses global variables

#    matchresult = {shapename:[0,0] for shapename in test_dist.keys()}
    matchresult = {}
    
    # test_id, train_id are int    
    
    for test_id in result.keys():
        catarr1 = test_cat[test_id] # the set of categories of a test sample
        # [low-level category,high-level category,higher-level category,...]
        
        if len(catarr1) <= matchorder:
            continue # test sample doesn't have that many categories
        matches = 0 # statistics
        trytomatch = 0 # statistics
        keytomatch = catarr1[matchorder]
        # a category, could be a low-level or high-level category
        
        if keytomatch not in train_dist:
            continue # no such category in the training set... why bother?
        
        for train_id in result[test_id]:
            trytomatch += 1
            catarr2 = train_cat[train_id] # the set of categories of a shape match
            if keytomatch in catarr2: # do their key also matches?
                matches += 1
        
        if train_dist[keytomatch] < trytomatch: # there are 5 shape matches 
            trytomatch += 1 # per sample but sometimes there aren't that
            # many shapes of that category in the database; don't over count
        
        if keytomatch in matchresult:
            matchresult[keytomatch][0] += matches # add to overall statistics
            matchresult[keytomatch][1] += trytomatch
        else:
            matchresult[keytomatch] = [matches,trytomatch]
    
    return matchresult
    
mr0 = checkmatch(0)
mr1 = checkmatch(1)
mr2 = checkmatch(2)
mr3 = checkmatch(3)
            
def overallmatchness(topN): # use global variables
    
    relretrlist = {shapename:[0.,0.,0.] for shapename in test_dist.keys()}
    for test_id in result.keys():
        for cat in test_cat[test_id]:
            relevant_and_retrieved = relretrlist[cat]
            relevant_and_retrieved[2] += topN # retrieved, fp + tp
            if cat in train_dist.keys():
                relevant_and_retrieved[1] += min(topN, train_dist[cat]) # fn + tp
            match = 0
            count = 0
            for train_id in result[test_id]:
                count += 1
                if count > topN:
                    break
                if cat in train_cat[train_id]:
                    match += 1
            relevant_and_retrieved[0] += min(match,topN) # relevant and retrieved, tp
            
    return relretrlist
    
def overallmatchness2(topN): # use global variables
    
    relretrlist = {shapename:[0.,0.,0.] for shapename in test_dist.keys()}
    for test_id in result.keys():
        for cat in test_cat[test_id]:
            relevant_and_retrieved = relretrlist[cat]
            relevant_and_retrieved[2] += topN # retrieved, fp + tp
            if cat in train_dist.keys():
                relevant_and_retrieved[1] += train_dist[cat] # fn + tp
            match = 0
            count = 0
            for train_id in result[test_id]:
                count += 1
                if count > topN:
                    break
                if cat in train_cat[train_id]:
                    match += 1
            relevant_and_retrieved[0] += min(match,topN) # relevant and retrieved, tp
            
    return relretrlist
    
r1 = overallmatchness(1)
#r2 = overallmatchness(2)
#r3 = overallmatchness(3)
#r4 = overallmatchness(4)
#r5 = overallmatchness(5)

f1 = 0
f1s = {}

def tof1(arr):
    precision = arr[0]/arr[2]
    recall = arr[0]/arr[1]
    return 2*precision*recall/(precision+recall)#[precision,recall]

def topr(arr):
    precision = arr[0]/arr[2]
    recall = arr[0]/arr[1]
    return precision,recall

for key in test_dist.keys():
    f1s[key] = []
for i in range(1,15):
    r = overallmatchness(i)
    for key in test_dist.keys():
        try:
            f1s[key].append(tof1(r[key]))
        except ZeroDivisionError:
            f1s[key].append(np.inf)

f1 = []
for i in range(1,20):
    r = overallmatchness(i)
    f1.append(tof1(np.sum(np.array(r.values()),axis=0)))

pr = []
for i in range(1,50):
    r = overallmatchness2(i)
    pr.append(topr(np.sum(np.array(r.values()),axis=0)))
    
a = np.argsort(np.array(test_dist_low.values()))
b = range(len(test_dist.keys())-10,len(test_dist.keys()))
most_occurance = [test_dist_low.keys()[a[i]] for i in b]
most_occurance = [m for m in most_occurance if m in mr0.keys()]

from matplotlib import pyplot as plt
counts = [mr0[key][1] for key in most_occurance]
hits = [mr0[key][0] for key in most_occurance]
plt.figure()
ind = np.arange(len(most_occurance))+0.2
width = 0.5
p1 = plt.bar(ind, counts, width, color='r')
p2 = plt.bar(ind, hits, width, color='b')

plt.ylabel('Count')
plt.title('The correctness of some frequent categories')
plt.xticks(ind + width/2., tuple(most_occurance))
plt.yticks(np.arange(0, max(counts), 10))
plt.legend((p2[0], p1[0]), ('hit', 'miss'))

plt.figure()
plt.plot(range(1,20),f1)
plt.title('"F1" score if the result is limited to top N')
plt.xlabel('N')
plt.ylabel('F1(precison at N, recall at N)')


index = 9
f1scatter = []
f1scatter_name = []
for name in f1s.keys():
    v = f1s[name]
    if v[index] < np.inf:
        f1scatter.append(v[index])
        f1scatter_name.append(name)

f1scatter = np.array(f1scatter)
sizearr = np.array([test_dist[name] for name in f1scatter_name])

fig, ax = plt.subplots()
ax.scatter(sizearr,f1scatter,s=50)

for i, txt in enumerate(f1scatter_name):
    ax.annotate(txt, (sizearr[i]+1,f1scatter[i]+0.01))
plt.title('"Score at 10 vs. occurance')
plt.xlabel('occurance in test set')
plt.ylabel('F1(precison at 10, recall at 10)')

pr_a = np.array(pr)
plt.figure()
plt.plot(pr_a[:,1],pr_a[:,0])
plt.title('precison vs. recall')
plt.xlabel('recall')
plt.ylabel('precison at N ')
