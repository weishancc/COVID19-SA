# -*- coding: utf-8 -*-
import pandas as pd

def preprocess(filepath, mode):
    assert mode in ['train', 'test']
    
    df = pd.read_csv(filepath)
    if mode == 'train':
        # sample 10% of data and not duplicate
        df_train = df.sample(frac = 0.01, random_state = None)
        
        #Select useful columns
        df_train = df_train.loc[:, ['title1_en', 'label']]
        df_train.rename(columns = {'title1_en':'Comment'}, inplace = True)
        
        # Map label and save tsv
        label_map = {'agreed': 0, 'disagreed': 1, 'unrelated': 2}
        df_train = df_train.replace({'label': label_map})
        df_train.to_csv('train.tsv', sep = '\t', index = False)     
        return df_train
    else:
        df_test = df.sample(frac = 0.01, random_state = None)
        df_test = df_test.loc[:, ['title1_en']]
        df_test.rename(columns = {'title1_en':'Comment'}, inplace = True)
        df_test.to_csv('test.tsv', sep = '\t', index = False) 
        return df_test   
    return None

