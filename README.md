Longitudinal Ensemble Integration leverages longitudinal multimodal data to build classifiers using a late fusion approach. In LEI, base predictors are trained on each modality over time before being ensembled at the late stage.

This repo implements a generalization of the original ensemble integration algorithm for longitudinal multimodal data (see ei directory). Yan Chak Li, Linhua Wang, Jeffrey N Law, T M Murali, Gaurav Pandey. Integrating multimodal data through interpretable heterogeneous ensembles, Bioinformatics Advances, Volume 2, Issue 1, 2022, vbac065, https://doi.org/10.1093/bioadv/vbac065.

We include scripts for executing different configurations of all steps of the LEI algorithm.
Implementation on ADNI/TADPOLE is contained in longitudinal_tadpole_v2 directory.
