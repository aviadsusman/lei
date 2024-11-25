import argparse
import os
import pickle as pkl
from eipy.full_lei import EnsembleIntegration
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# returns a flattened version of the total cohort and sequential version of final core cohort for row extraction
def get_data(col_thresh, row_thresh, longest_gap):
    dir_path =  f'longitudinal_tadpole_v2/data/{col_thresh}_{row_thresh}'
    total_cohort_name = 'total_cohort.pkl' 
    core_cohort_name = f'core_{longest_gap}.pkl'

    with open(os.path.join(dir_path, total_cohort_name), "rb") as file:
        total_cohort = pkl.load(file=file)
    with open(os.path.join(dir_path, core_cohort_name), "rb") as file:
        core_cohort_seqs = pkl.load(file=file)
    
    return total_cohort, core_cohort_seqs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Which data to use.")
    parser.add_argument('--col_thresh', type=int, help='Missingness threshold for filtering columns')
    parser.add_argument('--row_thresh', type=int, help='Missingness threshold for filtering rows')
    parser.add_argument('--longest_gap', type=int, help='Longest valid gap when constructing core cohort')
    parser.add_argument('--mode', action='store_true', help='Whether or not to include modality data in BP training')
    parser.add_argument('--split', type=int, help='EI random states will range from split to split+10')
    parser.add_argument('--sampling', type=str, default=None, help='EI random states will range from split to split+10')
    
    args = parser.parse_args()
    col_thresh = args.col_thresh
    row_thresh = args.row_thresh
    longest_gap = args.longest_gap
    with_mode = args.mode
    first_split = args.split
    sampling = args.sampling
    

    
    total_cohort, core_cohort_seqs = get_data(col_thresh=col_thresh, row_thresh=row_thresh, longest_gap=longest_gap)
    # for separating the core cohort from the modality data within the total cohort
    def core(row):
        if row['vis_info', 'RID'] in core_cohort_seqs.index:
            if row['vis_info', 'VISCODE'] in core_cohort_seqs[row['vis_info', 'RID']]:
                return True
            else:
                return False    
        else:
            return False
    core_cohort = total_cohort[total_cohort.apply(core, axis=1)].reset_index(drop=True)

    X = core_cohort.drop(columns = [('other', 'DX'), 'vis_info'])
    y = core_cohort['other','DX']

    if with_mode:
        mode_data = total_cohort[~total_cohort.apply(core, axis=1)].reset_index(drop=True)
        X_mode = mode_data.drop(columns = [('other', 'DX'), 'vis_info'])
        y_mode = mode_data['other','DX']
    else:
        X_mode = {mode: None for mode in X.columns.get_level_values(0).unique()} #so we can pass in X_mode independently of with_mode var
        y_mode = None

    for seed in range(first_split, first_split+10):
        EI = EnsembleIntegration(
        k_outer = 5,
        k_inner = 5,
        groups = core_cohort['vis_info', 'RID'],
        n_samples = 1,
        sampling_strategy = sampling,
        sampling_aggregation = None,
        n_jobs = -1,
        metrics = None, #multiclass defaults to macro f
        random_state = seed,
        parallel_backend = 'loky',
        project_name = f'{col_thresh} {row_thresh} {longest_gap} {with_mode}',
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

        for mode in X.columns.get_level_values(0).unique():
            EI.fit_base(X=X[mode], y=y, base_predictors=base_predictors, modality_name=mode, X_mode=X_mode[mode], y_mode=y_mode)
        

        has_mode = 'with_mode_data' if with_mode else 'no_mode_data'
        sample_strat = sampling if sampling!=None else 'no_sampling'

        results_path = f'longitudinal_tadpole_v2/bps/results/{sample_strat}/{col_thresh}_{row_thresh}_{longest_gap}_{has_mode}'
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        with open(os.path.join(results_path, f'{seed}.pkl'), 'wb') as file:
            pkl.dump(obj=EI, file=file)
        

