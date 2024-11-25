Longitudinal Ensemble Integration leverages longitudinal multimodal data to build classifiers using a late fusion approach. In LEI, base predictors are trained on each modality over time before being ensembled at the late stage.

This repo implements an augmentation of the original ensemble integration algorithm to be suitable for longitudinal and multimodal data (see ei directory). Yan Chak Li, Linhua Wang, Jeffrey N Law, T M Murali, Gaurav Pandey. Integrating multimodal data through interpretable heterogeneous ensembles, Bioinformatics Advances, Volume 2, Issue 1, 2022, vbac065, https://doi.org/10.1093/bioadv/vbac065.

We include scripts for executing the different configurations of the different steps of the LEI algorithm for reproduciblity.

Included in the stacking directory is an implementation of variable length sequence LSTMs in pytorch with a custom class for efficient time-distributed modeling on PackedSequence objects.
