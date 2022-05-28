# for visualisation using brat
#select admissions brat documents from the generated full brat files and cohort_doc.csv
#input an ORDO_ID + cohort_doc.csv + brat file prediction folder (e.g. docs_nlp+icd_all or doc_rd_pred)
#output a subfolder under the brat file prediction folder with the selected brat docs and annotations

import pandas as pd
import shutil
import os 
from pathlib import Path

#choose the rare disease as ORDO_ID and your selection criteria (NLP or ICD)
#ORDO_ID = 'Orphanet_791_Retinitis_pigmentosa'
#ORDO_ID = 'Orphanet_90062_Acute_liver_failure'
#ORDO_ID = 'Orphanet_3099_Rheumatic_fever'
#ORDO_ID = 'Orphanet_2302_Asbestos_intoxication'
ORDO_ID = 'Orphanet_391673_Necrotizing_enterocolitis'
#ORDO_ID = 'Orphanet_217260_Progressive_multifocal_leukoencephalopathy'
#ORDO_ID = 'Orphanet_280062_Calciphylaxis'
#ORDO_ID = 'Orphanet_209981_IRIDA_syndrome'
#ORDO_ID = 'Orphanet_3282_Multifocal_atrial_tachycardia'
#ORDO_ID = 'Orphanet_803_Amyotrophic_lateral_sclerosis'
criteria = 'NLP weak' # NLP strong, NLP weak, ICD-bp, ICD-NZ
brat_file_pred_folder_path='./doc_rd_pred/' #'./doc_rd_pred_MIMIC_III_DS_no_ICD_only_results/'

output_folder_path = './%s_%s_cohort' % (ORDO_ID, criteria.replace(' ', '_'))

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
        
column_name_criteria = 'ORDO ID by %s' % criteria

df_cohort_all = pd.read_csv('cohort_doc.csv')
df_cohort_all[column_name_criteria] = df_cohort_all[column_name_criteria].fillna('')
df_cohort_selected = df_cohort_all[df_cohort_all[column_name_criteria].str.contains(ORDO_ID)]
print(df_cohort_selected.head())
print(len(df_cohort_selected))

for filename_main in df_cohort_selected['doc-subject_ID-row_ID-rORDO_IDs_NLP_strong']:
    #construct full file path
    for extension in ['.txt','.ann']:
        file_name_os_path = Path(brat_file_pred_folder_path + filename_main + extension)
        if file_name_os_path.exists():
            shutil.copy(file_name_os_path,output_folder_path)
            
#copy brat configuration files
shutil.copy(Path(brat_file_pred_folder_path + 'annotation.conf'),output_folder_path)
shutil.copy(Path(brat_file_pred_folder_path + 'visual.conf'),output_folder_path)