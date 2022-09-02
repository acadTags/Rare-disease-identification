# Rare-disease-identification

This repository presents an approach using **ontologies** and **weak supervision** to identify rare diseases from clinical notes. The idea is illustrated below and the [data annotation](https://github.com/acadTags/Rare-disease-identification/tree/main/data%20annotation) for rare disease entity linking and ontology matching is available for download.

The latest preprint is available on arXiv, [Ontology-Based and Weakly Supervised Rare Disease Phenotyping from Clinical Notes](https://arxiv.org/abs/2205.05656). This is an extension of the [previous work](https://arxiv.org/abs/2105.01995) published in IEEE EMBC 2021.

## Entity linking and ontology matching
A graphical illustration of the entity linking and ontology matching process:
<p align="center">
    <img src="https://github.com/acadTags/Rare-disease-identification/blob/main/Graph%20representation.PNG" width=70% title="Ontology matching and entity linking for rare disease identification">
</p>

## Weak supervision (WS)
The process to create weakly labelled data with contextual representation is illustrated below:
<p align="center">
    <img src="https://github.com/acadTags/Rare-disease-identification/blob/main/Weak%20supervision%20illustrated.PNG" width=70% title="Weak supervision to improve entity linking">
</p>

## Rare disease mention annotations
The annotations of rare disease mentions created from this research are available in the folder [`data annotation`](https://github.com/acadTags/Rare-disease-identification/tree/main/data%20annotation).

## Implementation sources
* Main packages: See [`requirement.txt`](https://github.com/acadTags/Rare-disease-identification/blob/main/requirements.txt) (with conda scripts inside) for a full list. [BERT-as-service](https://bert-as-service.readthedocs.io/en/latest/) (follow guide to install), scikit_learn, Huggingface Transformers, numpy, nltk, gensim, pandasm, medcat, etc. 
* SemEHR can be installed from https://github.com/CogStack/CogStack-SemEHR
    * [Minimised SemEHR version](https://github.com/CogStack/CogStack-SemEHR/tree/safehaven_mini/installation) was used to process the MIMIC-III radiology reports.
* BlueBERT (Base, Uncased, PubMed+MIMIC-III) models are from https://github.com/ncbi-nlp/bluebert or https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12
* Ontology matching: 
    * ORDO to ICD-10 or UMLS https://www.ebi.ac.uk/ols/ontologies/ordo; 
    * ICD-10 to ICD-9 https://www.health.govt.nz/nz-health-statistics/data-references/mapping-tools/mapping-between-icd-10-and-icd-9; 
    * UMLS to ICD-9-CM https://bioportal.bioontology.org/ontologies/ICD9CM

## Reproducing results from the paper

This does not need to run the pipeline below, as it is based on the [`prediction scores`](https://github.com/acadTags/Rare-disease-identification/tree/main/data%20annotation/raw%20annotations%20(with%20model%20predictions)).

Move all the files inside `main_scripts` (and `other_scripts`) to the upper folder.

### Main results: Text-to-UMLS

MIMIC-III discharge summaries: `python step4_further_results_from_annotations.py`

MIMIC-III radiology reports: `python step4.1_further_results_from_annotations_for_rad.py`

Error analysis: `python error_analysis.py`

### Other results: UMLS-to-ORDO, Text-to-ORDO

UMLS-to-ORDO: calculated from results in `raw annotations (with model predictions)`.

Text-to-ORDO, mention-level: see `step7` and `step7.1` in `other_scripts`.

Text-to-ORDO, admission-level: see `step8` and `step8.1` in `other_scripts`.

## Pipeline

### Data and models
The data files and BERT models are placed according to the structure below. The SemEHR outputs for MIMIC-III discharge summaries (`mimic-semehr-smp-outputs\outputs`) and MIMIC-III radiology reports (`mimic-rad-semehr-outputs\outputs`) were obtained by running SemEHR.

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
└───models/
|   |   ... (phenotype confirmation model `.pik` files)
└───ontology/
|   |   ORDO2UMLS_ICD10_ICD9+titles_final_v2.xlsx 
        (ontology concept matching file) 
```

### Key pipeline scripts
* Weakly supervised data creation: `main_scripts/step1_tr_data_creat_ment_disamb.py`.
* Weakly supervised data representation and model training: `main_scripts/step3.4` for MIMIC-III discharge summaries, `main_scripts/step3.6` for MIMIC-III (and Tayside) radiology reports.
    - static BERT-based encoding is implemented in `def encode_data_tuple() in main_scripts/sent_bert_emb_viz_util.py` using BERT-as-service;
    - a fine-tuning approach with Huggingface Transformers is in `other_scripts/step3.8_fine_tune_bert_with_trainer.py`.

If all files are set (MIMIC-III data, SemEHR outputs, BERT models), the main steps of the whole pipeline can be run with `python run_main_steps.py`.

Note: This is mainly research-based implementation, rather than well-engineered software, but we hope that the code, data, and results provide more details to this work and are useful.

## Acknowledgement
This work has been carried out by members from [KnowLab](https://knowlab.github.io/), also thanks to the [EdiE-ClinicalNLP research group](https://www.ed.ac.uk/usher/clinical-natural-language-processing).

Acknowledgement to the icons used: 
* MIMIC icon from https://mimic.physionet.org/
* UMLS icon from https://uts.nlm.nih.gov/uts/umls/ 
* ORDO icon from http://www.orphadata.org/cgi-bin/index.php
* Ministry of Health, New Zealand icon from https://www.health.govt.nz/nz-health-statistics/data-references/mapping-tools/mapping-between-icd-10-and-icd-9
* ICD 10 icon from https://icd.who.int/browse10/Content/ICD10.png
