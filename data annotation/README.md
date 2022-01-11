## Data annotation

The 1,073 full rare disease mention annotations (from 312 MIMIC-III **discharge summaries**) are in [`full_set_RD_ann_MIMIC_III_disch.csv`](https://github.com/acadTags/Rare-disease-identification/blob/main/data%20annotation/full_set_RD_ann_MIMIC_III_disch.csv).

The data split:
* the first 400 rows are used for validation, [`validation_set_RD_ann_MIMIC_III_disch.csv`](https://github.com/acadTags/Rare-disease-identification/blob/main/data%20annotation/validation_set_RD_ann_MIMIC_III_disch.csv), and
* the last 673 rows are used for testing, [`test_set_RD_ann_MIMIC_III_disch.csv`](https://github.com/acadTags/Rare-disease-identification/blob/main/data%20annotation/test_set_RD_ann_MIMIC_III_disch.csv).

The 198 rare disease mention annotations (from 145 MIMIC-III **radiology reports**) are in [`test_set_RD_ann_MIMIC_III_disch.csv`](https://github.com/acadTags/Rare-disease-identification/blob/main/data%20annotation/test_set_RD_ann_MIMIC_III_rad.csv). To note that radiology reports were only used for testing and not for validation.

**To note**: a row can only be consider a true phenotype of the patient only when the value of the column **gold mention-to-ORDO label** is 1.

## Data sampling and annotation procedure
* (i) Randomly sampled 500 discharge summaries (and 1000 radiology reports) from MIMIC-III

* (ii) 312 of the 500 discharge summaries (and 145 of the 1000 radiology reports) have at least one positive UMLS mention linked to ORDO, as identified by SemEHR; there are altogether 1073 (and 198 in radiology reports) UMLS/ORDO mentions.

* (iii) 3 medical informatics researchers (staff or PhD students) annotated the 1,073 mentions (and 2 medical informatics researchers annotated the 198 mentions in radiology reports), regarding whether they are the correct patient phenotypes matched to UMLS and ORDO. Contradictions in the annotations were then resolved by another research staff having biomedical background.

## Data dictionary

| Column   Name                                | Description                                                                                                                                                                                                   |
|----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ROW_ID                                       | Identifier unique to each row, see [`https://mimic.physionet.org/mimictables/noteevents/`](https://mimic.physionet.org/mimictables/noteevents/)                                                                                                                                                     |
| SUBJECT_ID                                | Identifier unique to a patient, see [`https://mimic.physionet.org/mimictables/noteevents/`](https://mimic.physionet.org/mimictables/noteevents/)                                                                                                                                                                                                              |
| HADM_ID                                      | Identifier unique to a patient hospital stay, see [`https://mimic.physionet.org/mimictables/noteevents/`](https://mimic.physionet.org/mimictables/noteevents/)                                                                                                                                                                                                              |
| document structure name                    | The document structure name of the mention. The document structure name is identified by   SemEHR  (only for discharge summaries).                                                                                                          |
| document structure offset in full document | The start and ending offsets of the document structure texts (or template) in the whole discharge summary. The document structure is parsed by SemEHR with regular expressions  (only for discharge summaries).                            |
| mention                                      | The mention identified by SemEHR.                                                                                                                                                                          |
| mention offset in document structure       | The start and ending offsets of the mention in the document structure (only for discharge summaries).                                                                                                                                      |
| mention offset in full document            | The start and ending offsets of the mention in the whole discharge summary. They can be calculated by `document structure offset in full document` and `mention offset in document structure`.                                                                                     |
| UMLS with desc                               | The UMLS identified by SemEHR, corresponding to the mention.                                                                                                                                                |
| ORDO with desc                               | The ORDO matched to the UMLS, using the linkage in the ORDO ontology, see [`https://www.ebi.ac.uk/ols/ontologies/ordo/terms?iri=http%3A%2F%2Fwww.orpha.net%2FORDO%2FOrphanet_3325`](https://www.ebi.ac.uk/ols/ontologies/ordo/terms?iri=http%3A%2F%2Fwww.orpha.net%2FORDO%2FOrphanet_3325) as an example.          |
| gold mention-to-UMLS label                 | Whether the mention-UMLS pair indicate a correct phenotype of the patient (i.e. a positive mention that correctly matches to the UMLS concept), 1 if correct, 0 if not.                                 |
| gold UMLS-to-ORDO label                    | Whether the matching is correct from the UMLS concept to the ORDO concept, 1 if correct, 0 if not.                                                                                                          |
| gold mention-to-ORDO label                 | Whether the mention-ORDO triple indicates a correct phenotype of the patient, 1 if correct, 0 if not. This column is 1 if both the mention-to-UMLS label and the UMLS-to-ORDO label are 1, otherwise 0. |

**Note:**
* These manual annotations are by no means to be perfect. There are hypothetical mentions which were difficult for the annotators to make a decision. Also, they are based on the output of [`SemEHR`](https://github.com/CogStack/CogStack-SemEHR), which does not have 100% recall, so the annotations may not cover all rare diseases mentions from the sampled discharge summaries.
* In row 323 from the full set or the validation set, the mention `nph` is not in the document structure (due to error in mention extraction), thus the `gold mention-to-UMLS label` is `-1`.

