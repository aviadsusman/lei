from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from eipy.metrics import fmax_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef, fbeta_score, f1_score
import pandas as pd
import numpy as np
from eipy.full_lei import EnsembleIntegration
import pickle as pkl
import os
import argparse

def prep_data():
    parent_path = '/sc/arion/projects/pandeg01a/susmaa01/lei/longitudinal_tadpole/tadpole_data/tadpole_data_as_dfs'
    with open(f"{parent_path}/tadpole_data_time_imptn_norm_thrshld30_dfs.pickle", "rb") as file:
        data = pkl.load(file=file)
    with open(f"{parent_path}/tadpole_labels_time_imptn_norm_thrshld30_dfs.pickle", "rb") as file:
        labels = pkl.load(file=file)

    groups = np.concatenate([np.arange(labels.shape[0]) for t in range(labels.shape[1]-1)])

    modes = data['bl'].keys()
    X = {mode: None for mode in modes}
    for mode in modes:
        mode_df = pd.concat([data[t][mode].reset_index(drop=True) for t in data if t!='m36'])
        mode_df.reset_index(drop=True, inplace=True)

        vis = mode_df.index // labels.shape[0]
        vis_norm = (vis - vis.min()) / (vis.max() - vis.min())
        mode_df['VISCODE'] = vis_norm

        X[mode] = mode_df
    
    y = pd.concat([labels[col] for col in labels.columns if col!='m36'])

    return X, y, groups

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EI arguments.")
    parser.add_argument('--sampling', type=str, help='Sampling strategy.')
    parser.add_argument('--splits', type=int, help='Number of CV splits')
    args = parser.parse_args()
    sampling = args.sampling
    splits = args.splits
    
    X, y, groups = prep_data()
    
    for seed in range(splits):
        print(f"Generating BPs for split {seed+1} of {splits}")
        EI = EnsembleIntegration(
        k_outer = 5,
        k_inner = 5,
        groups = groups,
        n_samples = 1,
        sampling_strategy = sampling,
        sampling_aggregation = None,
        n_jobs = -1,
        metrics = None, #multiclass defaults to macro f
        random_state = seed,
        parallel_backend = 'loky',
        project_name = f'v1_cohort_{seed}',
        model_building = False,
        verbose = 1)

        base_predictors = {'ADAB': AdaBoostClassifier(algorithm='SAMME'),
                           'XGB': XGBClassifier(),
                           'DT': DecisionTreeClassifier(),
                           'RF': RandomForestClassifier(),
                           'GB': GradientBoostingClassifier(),
                           'KNN': KNeighborsClassifier(),
                           'LR': LogisticRegression(),
                           'NB': GaussianNB(),
                           'MLP': MLPClassifier(),
                           'SVM': SVC(probability=True)}

        for mode in X:
            EI.fit_base(X=X[mode], y=y, base_predictors=base_predictors, modality_name=mode)
        
        sample_strat = sampling if sampling!=None else 'no_sampling'
        results_path = f'longitudinal_tadpole_v2/bps/results/v1/{sample_strat}'
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        with open(os.path.join(results_path, f'{seed}.pkl'), 'wb') as file:
            pkl.dump(obj=EI, file=file)