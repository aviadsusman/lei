import pickle as pkl
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
import pandas as pd
from longitudinal_tadpole_v2.bps.bps import get_data
import numpy as np
from tqdm import tqdm
import torch
import os
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn.functional as F
import argparse
'''
File for transforming BP data into padded tensors with labels swapped.
'''

def recover_id_vis(ei):
    '''
    Use EI attributes to recover the sample IDs
    and time points of rows in BP data
    '''

    col_thresh = ei.project_name.split(" ")[0]
    row_thresh = ei.project_name.split(" ")[1]
    longest_gap = ei.project_name.split(" ")[2]
    
    total_cohort, core_cohort_seqs = get_data(col_thresh=col_thresh, row_thresh=row_thresh, longest_gap=longest_gap)
    
    def core(row):
        if row['vis_info', 'RID'] in core_cohort_seqs.index:
            if row['vis_info', 'VISCODE'] in core_cohort_seqs[row['vis_info', 'RID']]:
                return True
            else:
                return False    
        else:
            return False
    
    core_cohort = total_cohort[total_cohort.apply(core, axis=1)].reset_index(drop=True)
    ids = core_cohort['vis_info'].loc[:,('RID', 'VISCODE')]
    
    # y for stratification
    y = core_cohort['other', 'DX']
    
    # replicate nested CV on RID and VISCODE columns    
    ensemble_test_id = [ids.iloc[test_idx].reset_index(drop=True) for _, test_idx in ei.cv_outer.split(ids, y, groups=ei.groups)]
    ensemble_train_id = []
    for train_idx, _ in ei.cv_outer.split(ids, y, groups=ei.groups):
        id_o, y_o, groups_o = ids.iloc[train_idx], y.iloc[train_idx], ei.groups.iloc[train_idx]
        ensemble_train_id_fold = pd.concat([id_o.iloc[test_idx].reset_index(drop=True) for _, test_idx in ei.cv_inner.split(id_o, y_o, groups=groups_o)], axis=0)
        ensemble_train_id.append(ensemble_train_id_fold)
    
    return ensemble_train_id, ensemble_test_id

def get_seqs_list(df, ref, first_dx_as_feature):
    X_seqs_list = []
    y_seqs_list = []

    if first_dx_as_feature:
        num_classes = len(df.loc[:,'labels'].value_counts())

    for sample in ref['RID'].unique():
        sample_rows = ref[ref['RID']==sample].sort_values('VISCODE').index
        data = torch.tensor(df.loc[sample_rows].values)
        X, y = data[:-1,:-1], data[1:,-1] #row indexing gives proper labels for stacking

        if first_dx_as_feature:
            first_dx_tensor = torch.full((X.shape[0],1), data[0,-1] / num_classes)
            X = torch.cat((X, first_dx_tensor), dim=1)

        X_seqs_list.append(X)
        y_seqs_list.append(y)
    
    return X_seqs_list, y_seqs_list

def bp_tensors(dir_path, first_dx_as_feature=False):
    '''
    Loads in EI objects from specific
    directory and creates tensors out of BP data.
    Returns list the size of the directory containing CV splits.
    '''
    bp_data_as_tensors = []
    
    print(f"Retreiving padded sequences from base predictor data:")
    
    for filename in tqdm(os.listdir(dir_path)):
        with open(os.path.join(dir_path, filename), "rb") as file:
            ei = pkl.load(file)

        # gets the ids and viscodes of rows in ensemble_training_data 
        # and ensemble_test_data lists. lists of dfs.
        ensemble_train_ids, ensemble_test_ids = recover_id_vis(ei)
        
        ei_tensor_list = [] # length = ei.k_outer
        for fold in range(ei.k_outer):
            X_train, y_train = get_seqs_list(ei.ensemble_training_data[fold], ensemble_train_ids[fold], first_dx_as_feature=first_dx_as_feature)
            X_test, y_test = get_seqs_list(ei.ensemble_test_data[fold], ensemble_test_ids[fold], first_dx_as_feature=first_dx_as_feature)
            
            # Time independent dummy label for stratification. Dementia prevalence doesn't work because smallest class only has one member.
            # dem_prev = [torch.sum(tensor==2).item() for tensor in y_train]
            has_dem = [torch.any(tensor == 2).item() for tensor in y_train]
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=has_dem)

            data_dict = {'X_train': X_train,
                         'X_val': X_val,
                         'X_test': X_test,
                         'y_train': y_train,
                         'y_val': y_val,
                         'y_test': y_test}
            data_dict = {k : pad_sequence(v, batch_first=True, padding_value=float('-inf')) for k,v in data_dict.items()}
            data_dict['train lengths'] = torch.tensor([len(y) for y in y_train])
            data_dict['val lengths'] = torch.tensor([len(y) for y in y_val])
            data_dict['test lengths'] = torch.tensor([len(y) for y in y_test])

            
            ei_tensor_list.append(data_dict)
        bp_data_as_tensors.append(ei_tensor_list)

    # list the size of the directory 
    # whose entries are lists the size of length ei_from_file.k_outer
    # whose entries are dicts with data splits in the form of lists of variable length tensors
    return bp_data_as_tensors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load in data for processing")
    parser.add_argument('--path', type=str, help='Path to dir of data to be transformed')
    parser.add_argument('--v1', action='store_true', help='Different pipeline if for v1 cohort')
    args = parser.parse_args()
    dir_path = args.path

    data = bp_tensors(dir_path=dir_path)
    
    config_path = dir_path.split('results/')[1]
    parent_path = 'longitudinal_tadpole_v2/stacking/rnn_based/data/bps'
    full_dir = os.path.join(parent_path, config_path)
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)

    with open(os.path.join(full_dir, f'split_cv_tensors.pkl'), 'wb') as file:
        pkl.dump(obj=data, file=file)