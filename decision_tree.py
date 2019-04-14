#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
from pandas import DataFrame
import math
from collections import Counter
from pprint import pprint

df = DataFrame.from_csv('treedata.csv')
df.keys()[0]


# In[62]:


def entrope(probs):
    return sum( [-prob*math.log(prob, 2) for prob in probs] )

def entropy_lst(lst):
    cnt = Counter(c for c in lst)
    num_ins = len(lst)*1.0
    #print("number of instances are: ",num_ins)
    probs = [x/num_ins for x in cnt.values()]
    #print("classes ", min(cnt), max(cnt))
    #print("prob of class {0} is {1} ".format(min(cnt),min(probs)))
    #print(" prob of class {0} is {1} ".format(max(cnt),max(probs)))
    return entrope(probs)
    
#print("Input data set is: ", df['class'])
tot_ent = entropy_lst(df['class'])
print('Total entropy of class is: ', tot_ent)


# In[60]:


def info_gain(df, att_nm, tar_atnm, trace=0):
    print(att_nm)
    dflt = df.groupby(att_nm)
    #for name, grp in dflt:
    #    print("name ", name,"group ", grp)
    obs = len(df.index)*1.0
    #print("obs ",obs)
    dff_ag = dflt.agg({tar_atnm: [entropy_lst, lambda x: len(x)/obs]})[tar_atnm]
    #print("\n\n\ndff ag; ",dff_ag)
    dff_ag.columns = ['Entropy','PropObservations']
    new_intr = sum(dff_ag['Entropy']*dff_ag['PropObservations'])
    old_intr = entropy_lst(df[tar_atnm])
    return old_intr - new_intr
  
#print(df)
print('Info-gain for Outlook is :'+str( info_gain(df, 'Outlook', 'class')),"\n")
print('\n Info-gain for Humidity is: ' + str( info_gain(df, 'Humidity', 'class')),"\n")
print('\n Info-gain for Wind is:' + str( info_gain(df, 'Wind', 'class')),"\n")
print('\n Info-gain for Temperature is:' + str( info_gain(df, 'Temperature','class')),"\n")

print("............................................ID3 ALGO..........................")

def id3algo(df, tar_atnm, att_nm, default_class = None):
    cnt = Counter(x for x in df[tar_atnm])
    if len(cnt)==1:
        return next(iter(cnt))
    elif df.empty or (not att_nm):
        return default_class
    else:
        default_class =max(cnt.keys())
        gain = [info_gain(df, attr, tar_atnm) for attr in att_nm]
        ind_max = gain.index(max(gain))
        best_atr = att_nm[ind_max]
        
        tree = {best_atr:{}}
        remain_attr = [i for i in att_nm if i!=best_atr]
        for at_val,dtset in df.groupby(best_atr):
            subtree = id3algo(dtset, tar_atnm,remain_attr, default_class)
            tree[best_atr][at_val] = subtree
        return tree


# In[61]:


att_nm = list(df.columns)
att_nm.remove('class')

tree = id3algo(df, 'class', att_nm)
pprint(tree)
attribute = next(iter(tree))
print("best attribute ",attribute)
print("Tree keys ", tree[attribute].keys())


# In[ ]:





# In[ ]:




