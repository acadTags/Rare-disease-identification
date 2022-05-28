#create context window and other metadata for a mention identified by NLP (WS or SS) for annotation
#generates annotation files for each annotator and the k random samples for all annotators
#input: brat files of NLP identified cohort, selected by using step14_select_cohort_brat_documents.py under the main project folder
#output: (i) annotation files per ORDO ID, 
#        (ii) summary annotation files of all ORDO IDs
#        (iii) randomly sampled annotated data (with number of samples k=50, and random_state as 1234, as input)
#        (iv) splitted annotation files per annotators (with num_annotators=5 as input)

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import math

#ORDO_ID = 'Orphanet_791_Retinitis_pigmentosa'
#ORDO_ID = 'Orphanet_90062_Acute_liver_failure'
#ORDO_ID = 'Orphanet_3099_Rheumatic_fever'
#ORDO_ID = 'Orphanet_2302_Asbestos_intoxication'
#ORDO_ID = 'Orphanet_391673_Necrotizing_enterocolitis'
#ORDO_ID = 'Orphanet_217260_Progressive_multifocal_leukoencephalopathy'
#ORDO_ID = 'Orphanet_280062_Calciphylaxis'
#ORDO_ID = 'Orphanet_209981_IRIDA_syndrome'
#ORDO_ID = 'Orphanet_3282_Multifocal_atrial_tachycardia'
#ORDO_ID = 'Orphanet_803_Amyotrophic_lateral_sclerosis'
list_ORDO_ID = ['Orphanet_791_Retinitis_pigmentosa', 'Orphanet_90062_Acute_liver_failure', 'Orphanet_3099_Rheumatic_fever', 'Orphanet_2302_Asbestos_intoxication', 'Orphanet_391673_Necrotizing_enterocolitis', 'Orphanet_217260_Progressive_multifocal_leukoencephalopathy', 'Orphanet_280062_Calciphylaxis', 'Orphanet_209981_IRIDA_syndrome', 'Orphanet_3282_Multifocal_atrial_tachycardia', 'Orphanet_803_Amyotrophic_lateral_sclerosis']
window_character_size = 300

data2screen_all_ORDOs = []

for ORDO_ID in list_ORDO_ID:#tqdm(list_ORDO_ID):
    #print(ORDO_ID)
    dict_doc_to_data = {} # a dictionary from doc to the data to screen 
    for paradigm in ['weak','strong']:
        is_weak = paradigm == 'weak'
        is_strong = paradigm == 'strong'
        #print(is_weak,is_strong)
        path = './%s_NLP_%s_cohort/' % (ORDO_ID,paradigm)
        for filename in os.listdir(path):
            if filename.endswith('.ann'):
                with open(os.path.join(path,filename),encoding='utf-8') as f_content:
                    ann_doc = f_content.readlines()
                ann_doc = [x.strip() for x in ann_doc]
                #get mention positions
                for ann in ann_doc:
                    if ann[0] == 'T': # only process the entities              
                        ann_eles = ann.split('\t')
                        ann_id = ann_eles[0]
                        #print(ann_eles)
                        concept_with_pos_eles = ann_eles[1].split(' ')
                        #if len(concept_with_pos_eles) != 3:
                        #    print(concept_with_pos_eles)
                        if concept_with_pos_eles[0] == ORDO_ID:
                            #print(filename, concept_with_pos_eles)                
                            pos_start = int(concept_with_pos_eles[1])
                            if len(concept_with_pos_eles) == 3:
                                pos_end = int(concept_with_pos_eles[2])
                            elif len(concept_with_pos_eles) > 3:
                                pos_end = int(concept_with_pos_eles[-1])
                            
                            filename_doc = filename[:len(filename)-len('.ann')] + '.txt'
                            filename_id = filename.split('\\')[-1]
                            with open(os.path.join(path,filename_doc),encoding='utf-8') as f_content:
                                doc = f_content.read()
                                mention = doc[pos_start:pos_end]
                                text_extracted = doc[pos_start-window_character_size:pos_start] + '*****' + mention + '*****' + doc[pos_end:pos_end+window_character_size]
                                #print(text_extracted)
                                #doc_and_mention_pos = filename_id + '-' + str(pos_start) + '-' + str(pos_end)
                                if filename_id not in dict_doc_to_data:
                                    #print(is_weak, paradigm)
                                    dict_doc_to_data[filename_id] = [filename_id,is_weak,is_strong,ann_id,text_extracted,mention,ORDO_ID]
                                    #print(filename_id,is_weak)
                                else:
                                    #update everything
                                    _,is_weak_prev,is_strong_prev,ann_id_prev,text_extracted_prev,mention_prev,_ =  dict_doc_to_data[filename_id]
                                    is_weak_updated = is_weak or is_weak_prev
                                    is_strong_updated = is_strong or is_strong_prev
                                    if ann_id not in ann_id_prev.split(';'):
                                        ann_id = ann_id_prev + ';' + ann_id
                                        text_extracted = text_extracted_prev + '\n\n' + text_extracted
                                        mention = mention_prev + ';' + mention
                                    else:
                                        ann_id = ann_id_prev
                                        text_extracted = text_extracted_prev
                                        mention = mention_prev
                                    dict_doc_to_data[filename_id] = [filename_id,is_weak_updated,is_strong_updated,ann_id,text_extracted,mention,ORDO_ID]
    data2screen = list(dict_doc_to_data.values())
    #display statistics and output the sheet for the ORDO_ID
    df_for_screening = pd.DataFrame(data2screen,columns=['doc-subject_ID-row_ID-rORDO_IDs_NLP_strong','WS','SS','ann_ID','Text in context window','mention','ORDO with desc'])
    #print number for WS and SS
    print(ORDO_ID,'all:', len(df_for_screening),'WS:',len(df_for_screening[df_for_screening['WS']==True]),'SS:',len(df_for_screening[df_for_screening['SS']==True]))
    df_for_screening.to_excel('for screening %s.xlsx' % ORDO_ID,index=False)             
    
    data2screen_all_ORDOs = data2screen_all_ORDOs + data2screen

#display statistics and output a single sheet for all the ORDOs selected
df_for_screening_all_ORDOs = pd.DataFrame(data2screen_all_ORDOs,columns=['doc-subject_ID-row_ID-rORDO_IDs_NLP_strong','WS','SS','ann_ID','Text in context window','mention','ORDO with desc'])    
#print number for WS and SS
print('summary - all:', len(df_for_screening_all_ORDOs),'WS:',len(df_for_screening_all_ORDOs[df_for_screening_all_ORDOs['WS']==True]),'SS:',len(df_for_screening_all_ORDOs[df_for_screening_all_ORDOs['SS']==True]))
df_for_screening_all_ORDOs.to_excel('for screening 10 ORDOs.xlsx',index=False)

#generate files for each annotators
num_annotators = 5
num_data = len(df_for_screening_all_ORDOs)
n = math.ceil(num_data / num_annotators)
id_ann = 0
for g, df_split_for_ann in df_for_screening_all_ORDOs.groupby(np.arange(num_data) // n):
    id_ann = id_ann + 1
    print(df_split_for_ann.shape)
    df_split_for_ann.to_excel('rare disease cases by NLP for screening - ann%d.xlsx' % id_ann)

#k randomly sampled data for all annotators
k=50
df_for_screening_all_ORDOs_sampled = df_for_screening_all_ORDOs.sample(k,random_state=1234)
print('sampled - all:', len(df_for_screening_all_ORDOs_sampled),'WS:',len(df_for_screening_all_ORDOs_sampled[df_for_screening_all_ORDOs_sampled['WS']==True]),'SS:',len(df_for_screening_all_ORDOs_sampled[df_for_screening_all_ORDOs_sampled['SS']==True]))
df_for_screening_all_ORDOs_sampled.to_excel('for screening 10 ORDOs - %d sampled.xlsx' % k,index=False)

#console output
'''
Orphanet_791_Retinitis_pigmentosa all: 194 WS: 183 SS: 55
Orphanet_90062_Acute_liver_failure all: 213 WS: 213 SS: 103
Orphanet_3099_Rheumatic_fever all: 271 WS: 271 SS: 268
Orphanet_2302_Asbestos_intoxication all: 164 WS: 164 SS: 155
Orphanet_391673_Necrotizing_enterocolitis all: 117 WS: 117 SS: 116
Orphanet_217260_Progressive_multifocal_leukoencephalopathy all: 38 WS: 32 SS: 36
Orphanet_280062_Calciphylaxis all: 30 WS: 30 SS: 24
Orphanet_209981_IRIDA_syndrome all: 229 WS: 229 SS: 194
Orphanet_3282_Multifocal_atrial_tachycardia all: 110 WS: 110 SS: 102
Orphanet_803_Amyotrophic_lateral_sclerosis all: 62 WS: 62 SS: 58
summary - all: 1428 WS: 1411 SS: 1111
(286, 7)
(286, 7)
(286, 7)
(286, 7)
(284, 7)
sampled - all: 50 WS: 50 SS: 40
'''