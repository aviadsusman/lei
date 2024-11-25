import numpy as np
import pandas as pd

from longitudinal_tadpole_v2.stacking.bps_to_tensors import recover_id_vis

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import argparse
import os
from tqdm import tqdm
import pickle as pkl
from copy import deepcopy

def swap_labels(ensemble_data, train_test, fold, rid=False, vis=False):
    train_test_map = {'train': 0, 'test': 1}
    ids = recover_id_vis(ei)[train_test_map[train_test]][fold].reset_index(drop=True)
    ids.columns = pd.MultiIndex.from_product([ids.columns, [''],[''],['']], names=ensemble_data.columns.names)

    data_with_ids = pd.concat([ensemble_data, ids], axis=1)
    data_with_ids = data_with_ids.groupby('RID').agg(list)

    def sort_by_vis(row):
        sort_order = np.argsort(row['VISCODE','','',''])
        return row.apply(lambda x: [x[i] for i in sort_order])
    data_ids_sorted = data_with_ids.apply(lambda row: sort_by_vis(row), axis=1)

    data_ids_sorted['labels'] = data_ids_sorted['labels'].apply(lambda x: x[1:])

    data_ids_sorted.loc[:, data_ids_sorted.columns.drop('labels')] = data_ids_sorted.loc[:, data_ids_sorted.columns.drop('labels')].map(lambda x: x[:-1])

    data_labels_swapped = data_ids_sorted.explode(list(data_ids_sorted.columns)).reset_index()

    if rid:
        data_labels_swapped['RID'] = (data_labels_swapped['RID'] - data_labels_swapped['RID'].min()) /(data_labels_swapped['RID'].max() - data_labels_swapped['RID'].min())
    else:
        data_labels_swapped = data_labels_swapped.drop(columns=['RID'])
    
    if vis:
        data_labels_swapped['VISCODE'] = (data_labels_swapped['VISCODE'] - data_labels_swapped['VISCODE'].min()) /(data_labels_swapped['VISCODE'].max() - data_labels_swapped['VISCODE'].min())
    else:
        data_labels_swapped = data_labels_swapped.drop(columns=['VISCODE'])

    data_labels_swapped = data_labels_swapped.astype({col: 'float' for col in data_labels_swapped.columns if col != 'labels'})
    data_labels_swapped['labels'] = data_labels_swapped['labels'].astype('int')
    
    return data_labels_swapped

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Which bps to use.")
    parser.add_argument('--dir', type=str, help='Directory of bps')
    parser.add_argument('--rid', action='store_true', help='Whether to include patient id as bp feature')
    parser.add_argument('--vis', action='store_true', help='Whether to include viscode as bp feature')
    args = parser.parse_args()
    dir_path = args.dir
    rid = args.rid
    vis = args.vis
    
    for filename in tqdm(os.listdir(dir_path)):
        with open(os.path.join(dir_path, filename), "rb") as file:
            ei = pkl.load(file)
        
        og_train = deepcopy(ei.ensemble_training_data)
        og_test = deepcopy(ei.ensemble_test_data)
        for fold, data in enumerate(ei.ensemble_training_data):
            ei.ensemble_training_data[fold] = swap_labels(ensemble_data=data, train_test='train', fold=fold, rid=rid, vis=vis)
        for fold, data in enumerate(ei.ensemble_test_data):
            ei.ensemble_test_data[fold] = swap_labels(ensemble_data=data, train_test='test', fold=fold, rid=rid, vis=vis)
        
        ensemble_predictors = {
                            'S.ADAB': AdaBoostClassifier(),
                            'S.XGB': XGBClassifier(),
                            'S.DT': DecisionTreeClassifier(),
                            "S.RF": RandomForestClassifier(),
                            'S.GB': GradientBoostingClassifier(),
                            'S.KNN': KNeighborsClassifier(),
                            'S.LR': LogisticRegression(),
                            'S.NB': GaussianNB(),
                            'S.MLP': MLPClassifier(),
                            'S.SVM': SVC(probability=True),
        }

        ei.fit_ensemble(ensemble_predictors=ensemble_predictors)

        results_dict = {'ei': ei, 
                        'original ensemble training data': og_train, 
                        'original ensemble test data': og_test}
        
        parent_path = 'longitudinal_tadpole_v2/stacking/static_ei/results'
        child_path = dir_path.split('bps/')[1]
        results_path = os.path.join(parent_path, child_path)
        has_rid = '_rid' if rid else ''
        has_vis = '_vis' if vis else ''

        if not os.path.exists(results_path):
            os.makedirs(results_path)
        with open(f'{results_path}/{ei.random_state}{has_rid+has_vis}.pkl', 'wb') as file:
            pkl.dump(obj=results_dict, file=file)