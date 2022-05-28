# Rare-disease-identification

This repository presents an approach using **ontologies** and **weak supervision** to identify rare diseases from clinical notes. The idea is illustrated below and the [data annotation](https://github.com/acadTags/Rare-disease-identification/tree/main/data%20annotation) for rare disease entity linking and ontology matching is available for download.

The [latest preprint](https://arxiv.org/abs/2205.05656) of this work is available on arXiv (including the [supplementary material](https://github.com/acadTags/Rare-disease-identification/blob/main/Supplementary%20Material%20for%20%22Ontology-Based%20and%20Weakly%20Supervised%20Rare%20Disease%20Phenotyping%20from%20Clinical%20Notes%22.pdf) in the repository). This is an extension of the [previous work](https://arxiv.org/abs/2105.01995) published in IEEE EMBC 2021.

## Entity linking and ontology matching
A graphical illustration of the entity linking and ontology matching process:
<p align="center">
    <img src="https://github.com/acadTags/Rare-disease-identification/blob/main/Graph%20representation.PNG" width=80% title="Ontology matching and entity linking for rare disease identification">
</p>

## Weak supervision (WS)
The process to create weakly labelled data with contextual representation is illustrated below:
<p align="center">
    <img src="https://github.com/acadTags/Rare-disease-identification/blob/main/Weak%20supervision%20illustrated.PNG" width=80% title="Weak supervision to improve entity linking">
</p>

## Rare disease mention annotations
The annotations of rare disease mentions created from this research are available in the folder [`data annotation`](https://github.com/acadTags/Rare-disease-identification/tree/main/data%20annotation).

## Implementation sources
* Main packages: See `requirement.txt` for a full list. [BERT-as-service](https://bert-as-service.readthedocs.io/en/latest/) (follow guide to install), scikit_learn, Huggingface Transformers, numpy, nltk, gensim, pandasm, medcat, etc. 
* SemEHR can be installed from https://github.com/CogStack/CogStack-SemEHR
* BlueBERT (Base, Uncased, PubMed+MIMIC-III) models are from https://github.com/ncbi-nlp/bluebert or https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12
* Ontology matching: 
    * ORDO to ICD-10 or UMLS https://www.ebi.ac.uk/ols/ontologies/ordo; 
    * ICD-10 to ICD-9 https://www.health.govt.nz/nz-health-statistics/data-references/mapping-tools/mapping-between-icd-10-and-icd-9; 
    * UMLS to ICD-9-CM https://bioportal.bioontology.org/ontologies/ICD9CM

## Pipeline
The data files and BERT models are placed according to the structure below. The SemEHR outputs for MIMIC-III discharge summaries (`mimic-semehr-smp-outputs\outputs`) and MIMIC-III radilogy reports (`mimic-rad-semehr-outputs\outputs`) were obtained by running SemEHR.

```
└───bert-models
|   |   run_get_bluebert.sh
|   |   NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12
|   |   |   ... (model files)
└───data/
|   |   NOTEEVENTS.csv (from MIMIC-III)
|   |   DIAGNOSES_ICD.csv (from MIMIC-III)
|   |   PROCEDURES_ICD.csv (from MIMIC-III)
|   |   mimic-semehr-smp-outputs
|   |   |   outputs
|   |   |   |   ... (SemEHR output files of MIMIC-III DS)
|   |   mimic-rad-semehr-outputs
|   |   |   outputs
|   |   |   |   ... (SemEHR output files of MIMIC-III rad)
```

* Weakly supervised data creation: `step1_tr_data_creat_ment_disamb.py`.
* Weakly supervised data representation and model training: `step3.4` for MIMIC-III discharge summaries, `step3.6` for MIMIC-III (and Tayside) radiology reports.

If all files are set (bert-models, MIMIC-III data, SemEHR outputs), the main steps of the whole pipeline can be run with `python run_main_steps.py`.

## Reproducing main results from the paper

### Main results: Text-to-UMLS

MIMIC-III discharge summaries: `python step4_further_results_from_annotations.py`

MIMIC-III radiology reports: `python step4.1_further_results_from_annotations_for_rad.py`

Error analysis: `python error_analysis.py`

### Other results: UMLS-to-ORDO, Text-to-ORDO

UMLS-to-ORDO: calculated from results in `raw annorations (with model predictions)`.

Text-to-ORDO, mention-level: see `step7` and `step7.1` in `other_scripts`.

Text-to-ORDO, admission-level: see `step8` and `step8.1` in `other_scripts`.

## Acknowledgement
This work has been carried out by members from [KnowLab](https://knowlab.github.io/), also thanks to the [EdiE-ClinicalNLP research group](https://www.ed.ac.uk/usher/clinical-natural-language-processing).

Acknowledgement to the icons used: 
* MIMIC icon from https://mimic.physionet.org/
* UMLS icon from https://uts.nlm.nih.gov/uts/umls/ 
* ORDO icon from http://www.orphadata.org/cgi-bin/index.php
* Ministry of Health, New Zealand icon from https://www.health.govt.nz/nz-health-statistics/data-references/mapping-tools/mapping-between-icd-10-and-icd-9
* ICD 10 icon from https://icd.who.int/browse10/Content/ICD10.png
