# Rare-disease-identification

This repository presents an approach using **ontologies** and **weak supervision** to identify rare diseases from clinical notes. The idea is illustrated below and the [data annotation](https://github.com/acadTags/Rare-disease-identification/tree/main/data%20annotation) for rare disease entity linking and ontology matching is available for download.

The [preprint](https://arxiv.org/abs/2105.01995) of this work is on arXiv.

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
* SemEHR can be installed from https://github.com/CogStack/CogStack-SemEHR
* BlueBERT (Base, Uncased, PubMed+MIMIC-III) models are from https://github.com/ncbi-nlp/bluebert or https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12
* Ontology matching: 
    * ORDO to ICD-10 or UMLS https://www.ebi.ac.uk/ols/ontologies/ordo; 
    * ICD-10 to ICD-9 https://www.health.govt.nz/nz-health-statistics/data-references/mapping-tools/mapping-between-icd-10-and-icd-9; 
    * UMLS to ICD-9-CM https://bioportal.bioontology.org/ontologies/ICD9CM

## Acknowledgement
This work was carried out by members from [KnowLab](https://knowlab.github.io/), also thanks to the [EdiE-ClinicalNLP research group](https://www.ed.ac.uk/usher/clinical-natural-language-processing).

Acknowledgement to the icons used: 
* MIMIC icon from https://mimic.physionet.org/
* UMLS icon from https://uts.nlm.nih.gov/uts/umls/ 
* ORDO icon from http://www.orphadata.org/cgi-bin/index.php
* Ministry of Health, New Zealand icon from https://www.health.govt.nz/nz-health-statistics/data-references/mapping-tools/mapping-between-icd-10-and-icd-9
* ICD 10 icon from https://icd.who.int/browse10/Content/ICD10.png
