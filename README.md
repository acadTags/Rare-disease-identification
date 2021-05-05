# Rare-disease-identification

This repository presents an approach using **ontologies** and **weak supervision** to identify rare diseases from clinical notes. The idea is illustrated below and the data annotation is available for download.

## Entity linking and ontology matching
A graphical illustration of the entity linking and ontology matching process:
<p align="center">
    <img src="https://github.com/acadTags/Rare-disease-identification/blob/main/Graph%20representation.PNG" width=75% title="Ontology matching and entity linking for rare disease identification">
</p>

## Weak supervision (WS)
The process to create weakly labelled data with contextual representation is illustrated below:
<p align="center">
    <img src="https://github.com/acadTags/Rare-disease-identification/blob/main/Weak%20supervision%20illustrated.PNG" width=75% title="Weak supervision to improve entity linking">
</p>

## Rare disease annotation
The annotation of rare disease mentions created from this research is available at [link](https://github.com/acadTags/Rare-disease-identification/blob/main/Rare%20disease%20mention%20annotations%20from%20a%20sample%20of%20MIMIC-III%20discharge%20summaries.xlsx) as a `.xlsx` file; the description of the data is in the second sheet of the file.

**Note**: This annotation is by no means a perfect one, although annotated by 4 researchers in (bio-)medical informatics; also, it is based on the output of [SemEHR](https://github.com/CogStack/CogStack-SemEHR) so it may not cover all rare diseases mentions from the sampled discharge summaries.

## Acknowledgement
Acknowledgement to the icons used: 
* MIMIC icon from https://mimic.physionet.org/
* UMLS icon from https://uts.nlm.nih.gov/uts/umls/ 
* ORDO icon from http://www.orphadata.org/cgi-bin/index.php
* Ministry of Health, New Zealand icon from https://www.health.govt.nz/nz-health-statistics/data-references/mapping-tools/mapping-between-icd-10-and-icd-9
* ICD 10 icon from https://icd.who.int/browse10/Content/ICD10.png
