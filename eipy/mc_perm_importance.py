import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import copy
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class MultiClassPermutationImportance:
    def __init__(self, estimator, X, y, random_state, scoring, n_repeats=10, n_jobs=-1):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.random_state = random_state
        self.scoring = scoring
        self.score = self.scoring(self.estimator.predict(self.X), self.y)
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs

        self.importances_mean = None
        self.importances_std = None
    
    
    def mc_permutation_importance(self):
        bps = list(set([col[:-1] for col in self.X.columns]))
        with Parallel(n_jobs=self.n_jobs, verbose=0, backend='loky') as parallel:
            scores = parallel(
                delayed(self._permute_n_score)(bp=base_predictor, random_state=n)
                for base_predictor in bps
                for n in range(self.n_repeats))
        self.importances_mean = np.array(scores).reshape(-1, self.n_repeats).mean(axis=1)
        self.importances_std = np.array(scores).reshape(-1, self.n_repeats).std(axis=1)
    
    def _permute_n_score(self, bp, random_state):
        X_copy = copy.deepcopy(self.X)
        X_copy[bp] = X_copy[bp].sample(frac=1, random_state=random_state+self.random_state).to_numpy()
        
        return self.score - self.scoring(self.estimator.predict(X_copy), self.y)