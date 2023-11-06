# GENEX: Explainable Batch effect removal for Single-Cell RNA Sequencing Data

**Abstract**: Batch effect correction remains a major challenge in single-cell RNA-sequencing data integration despite the number of methods developed. Most existing methods are unable to fully deconvolute technical and biological variations. Here, we present GENEX that leverages a disentangling generative adversarial net-work to distinguish batch effects and technical variations from biological differences. Moreover, unlike most other batch correction methods, GENEX returns batch corrected gene expression values that pre-serve the original distribution and can be used for downstream analyses.  Additionally, GENEX incorpo-rates an explainable AI component to enable the identification of genes specifically associated with vari-ous covariates such as gender, disease state, and sequencing platform. We evaluated GENEX on both simulated and experimental data.  It outperformed other existing methods in terms of batch correction and biological signal preservation. On the simulated dataset, GENEX reliably recapitulated the ground truth batch-invariant gene expressions. It also identified differentially expressed genes more accurately compared to differential expression analysis using unintegrated data or integrated values returned by oth-er methods. On a peripheral blood mononuclear cell dataset, GENEX identified genes highly influenced by different sequencing platforms. Furthermore, applied to a COVID dataset, GENEX revealed cell type-specific gene signatures associated with disease severity.

## Installation of the dependency library:

To test and run the code please install the exported Conda environment:

`conda env create -f environment.yml`

This will create a new Conda environment with the same packages and versions as the original environment

## GENEX model running and Analysis

All the running pipeline and analysis are written in Jupyter Notebook.

## Citation for Dataset

PBMC:
1. [Tran, H.T.N., Ang, K.S., Chevrier, M. et al. "A benchmark of batch-effect correction methods for single-cell RNA sequencing data." Genome Biol 21, 12 (2020).](https://doi.org/10.1186/s13059-019-1850-9)

COVID:

2. [Li, Mengwei, et al. "DISCO: a database of Deeply Integrated human Single-Cell Omics data." Nucleic acids research 50.D1 (2022): D596-D602.](https://academic.oup.com/nar/article/50/D1/D596/6430491?login=false)

## Follow us on our Github
[JinmiaoChenLab Github Repo](https://github.com/JinmiaoChenLab)