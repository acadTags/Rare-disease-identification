# process the MIMIC-III NOTEEVENTS with SemEHR outputs and link to ORDO, ICD9 for coding comparison, also generate samples for validation
# input: some settings, NOTEEVENTS.csv, SemEHR processed MIMIC-III clinical notes
# output: (i) dataframe (.pik) and (ii) excel sheet containing UMLS, ORDO, ICD9 codes identified from discharge summaries; (iii) sampled data for validation regarding the mention-to-UMLS matching by SemEHR and UMLS-to-ORDO matching from ORDO ontology
# code adapted from chunk_test.py by Minhong
# last edit: 21 Oct 2021

import pandas as pd
import os
from tqdm import tqdm
import requests
import csv
from constants import MIMIC_3_DIR, SemEHR_DIR, SemEHR_DIR_rad, SemEHR_DIR_tayside_full
import json
from mimic3_data_processing_util import get_code_from_url,mergeDict,get_rare_disease_umls
from collections import defaultdict
import re
import random
import pickle
import sys
import argparse

parser = argparse.ArgumentParser(description="step1 - process the MIMIC-III NOTEEVENTS with SemEHR outputs and link to ORDO, ICD9 for coding comparison, also generate samples for validation")
parser.add_argument('-dc','--data-category', type=str,
                    help="category of data", default='Discharge summary')
args = parser.parse_args()

# 0. settings
#data_category = 'Radiology' # ''Discharge summary'' or 'Radiology'
#set the SemEHR output json file directory path 
if args.data_category == 'Discharge summary':
    SemEHR_DIR_path = SemEHR_DIR
elif args.data_category == 'Radiology':
    SemEHR_DIR_path = SemEHR_DIR_rad
#elif args.data_category == 'Radiology_Tayside': # using the full/orig Tayside brain imaging report # for this, see step0.1_tayside_data_processing
#    SemEHR_DIR_path = SemEHR_DIR_tayside_full
else:
    print('args.data_category unrecognised: %s' % args.data_category)
    sys.exit(0)
    
# set percentage threshold to filter out "frequent" matched rare disease UMLSs - here we don't do filtering, but give an option.
prevalence_percentage_threshold = 1#0.005 # this should be the prevalence of rare disease in the ICU-admitted patients. 
# Prevalence of rare disease in the US is about 7.5/10000, see [1] J. Textoris and M. Leone, ‘Genetic Aspects of Uncommon Diseases’, in Uncommon Diseases in the ICU, M. Leone, C. Martin, and J.-L. Vincent, Eds. Cham: Springer International Publishing, 2014, pp. 3–11.
SemEHR_in_text_matching_len_threshold = 0#3 # to avoid wrong SemEHR UMLS matching due to meanings in abbreviations and short length

# for selecting only first k rows
n_rows_selected = None #None #1000, 5000, None

# weather to generate data for validation
genData2valid = True
# number of random doc samples for validation
num_samples_valid = 500 #500 for discharge summaries in the study

# avoid ... in showing long sequence
pd.set_option("display.max_colwidth", 10000)

# csv file name
filename = os.path.join(MIMIC_3_DIR,'NOTEEVENTS.csv')

# 1. match manual ICD codes and their descriptions to discharge summaries

# take chunk of noteevents.csv as example
temp = pd.read_csv(filename,nrows=n_rows_selected)

# select only the discharge summaries or the radiology reports
temp = temp[temp['CATEGORY']==args.data_category]
print('temp:',temp)
# # only select the inpatient if radiology - only inpatients have ICD codes
# if args.data_category == 'Radiology':
    # temp = temp[temp['HADM_ID'] != '']

map_diagnoses_icd = pd.read_csv(os.path.join(MIMIC_3_DIR,'DIAGNOSES_ICD.csv'))
#map_diagnoses_icd.dtypes
#map_diagnoses_icd.shape

map_d_icd_diagnoses = pd.read_csv(os.path.join(MIMIC_3_DIR,'D_ICD_DIAGNOSES.csv'))
#d_icd_diagnoses.dtypes

# merge diagnoses_icd and d_icd_diagnoses by icd9_code (many to one, left outer join)
diagnoses_icd_ver2 = pd.merge(map_diagnoses_icd, map_d_icd_diagnoses, how="left", on=['ICD9_CODE'])
diagnoses_icd_ver2.shape

# we don't need row_id in these two data sets
# also don't need SEQ_NUM? Do we want it?
diagnoses_icd_ver2 = diagnoses_icd_ver2.drop(['ROW_ID_x','SEQ_NUM','ROW_ID_y'], axis=1)

# group diagnoses_icd_ver2 (key = subject_id & hadm_id)
diagnoses_icd_ver2['ICD9_CODE'] = diagnoses_icd_ver2['ICD9_CODE'].astype(str)
diagnoses_icd_ver2['SHORT_TITLE'] = diagnoses_icd_ver2['ICD9_CODE'] + ":" + diagnoses_icd_ver2['SHORT_TITLE'].astype(str)
#print(diagnoses_icd_ver2['SHORT_TITLE'])
diagnoses_icd_ver2['LONG_TITLE'] = diagnoses_icd_ver2['ICD9_CODE'] + ":" + diagnoses_icd_ver2['LONG_TITLE'].astype(str)

diagnoses_icd_ver3 = diagnoses_icd_ver2.groupby(["SUBJECT_ID","HADM_ID"],
                                                as_index=False)["ICD9_CODE","SHORT_TITLE","LONG_TITLE"].agg(lambda x:list(x)) # nice code
                                                
# merge the icd results to noteevents
df = pd.merge(temp,diagnoses_icd_ver3,on=["SUBJECT_ID","HADM_ID"],how='left')
for col_name in ["ICD9_CODE","SHORT_TITLE","LONG_TITLE"]:
    df[col_name] = df[col_name].fillna("")
print(df)

df = df.drop(['CHARTDATE','CHARTTIME','STORETIME','CGID','ISERROR'],axis=1)

# # test df statistics
#df.to_excel('df-test.xlsx',index=False)
# print(len(df.index))

# 2. retrieve rare disease UMLS from SemEHR output

# load a list of selected rare disease UMLS from ORDO
#rare_disease_umls_list = get_rare_disease_umls()
#dict_rare_disease_umls = dict((el,0) for el in rare_disease_umls_list)
dict_rare_disease_umls = get_rare_disease_umls()

#df['icd9'] = ""
#df['icd9'] = df['icd9'].apply(list)
#df['text_snippets'] = ""
#df['text_snippets'] = df['text_snippets'].apply(list)
df['doc_structure'] = "" # the document structure where the mention appeared
df['doc_structure'] = df['doc_structure'].apply(list)
#add offsets if the data category is radiology reports or others, fpr in disch the offsets are within a document structure.
#offsets for discharge summaries can be derived from here as well (but we didn't)
if args.data_category != 'Discharge summary':
    df['mention_offsets'] = ""
    df['mention_offsets'] = df['mention_offsets'].apply(list)
df['umls_RD'] = ""
df['umls_RD'] = df['umls_RD'].apply(list)
df['umls_RD;doc_structure;text_snippet_full;in_text;label'] = ""
df['umls_RD;doc_structure;text_snippet_full;in_text;label'] = df['umls_RD;doc_structure;text_snippet_full;in_text;label'].apply(list)

#dict to collect the filtered out umls annotations - cui_in-text_pref-label:frequency in all doc ann
dict_unselected_ann_freq = defaultdict(int)

for index, row in tqdm(df.iterrows()):
    #print(row['c1'], row['c2'])
    row_id = str(row['ROW_ID'])
    subj_id = str(row['SUBJECT_ID'])
    text_full = str(row['TEXT'])
    if args.data_category == 'Discharge summary':
        json_f_name = '%s/%s/%s.json' % (SemEHR_DIR_path, subj_id[0], subj_id + '_' + row_id)
    else:
        assert args.data_category == 'Radiology'
        json_f_name = '%s/doc-rad-%s-%s.json' % (SemEHR_DIR_path, subj_id, row_id)    
    try:
        with open(json_f_name) as json_file:
            data = json.load(json_file)
            
            #dict_icd_code={}
            dict_umls_code_per_row=defaultdict(int)
            if args.data_category == 'Discharge summary':
                # for discharge summaries - parsing the json format which has document structure information            
                for doc_part in data: # looping over doc structure type
                    doc_part_retrieved = data.get(doc_part,None)
                    
                    #extract all ICD9 codes
                    if doc_part_retrieved != None:
                        text_paragraph = doc_part_retrieved["text"] # the free texts of the document structure
                        doc_part_anns = doc_part_retrieved['anns']
                        #print(doc_part_anns)
                        doc_part_start_pos = int(doc_part_retrieved['start'])
                    else:
                        continue
                    
                    for doc_part_ann in doc_part_anns:    
                        doc_part_ann_features = doc_part_ann["features"]
                        
                        # filtering based on the annotation features
                        
                        #if doc_part_ann_features["Negation"] == "Affirmed" and doc_part_ann_features["STY"] == "Disease or Syndrome" and doc_part_ann_features["Experiencer"] == "Patient" and doc_part_ann_features["Temporality"] == "Recent": # with temporality as recent
                        if doc_part_ann_features["Negation"] == "Affirmed" and doc_part_ann_features["STY"] == "Disease or Syndrome" and doc_part_ann_features["Experiencer"] == "Patient": # 2020.11.2
                            in_text_ann = doc_part_ann_features["string_orig"]
                            umls_code = doc_part_ann_features["inst"]
                            umls_label = doc_part_ann_features["PREF"]
                            
                            #get positions
                            ann_start_ = doc_part_ann['startNode']
                            ann_start_pos = ann_start_['offset']
                            ann_end_ = doc_part_ann['endNode']
                            ann_end_pos = ann_end_['offset']
                            ann_relative_start_pos = ann_start_pos - doc_part_start_pos
                            ann_relative_end_pos = ann_end_pos - doc_part_start_pos
                            
                            #get text snippet: the part of text that contain the string mention identified from SemEHR
                            #text_snippet = text_paragraph[ann_relative_start_pos-200:ann_relative_start_pos] + '<' + text_paragraph[ann_relative_start_pos:ann_relative_end_pos] + '>' + text_paragraph[ann_relative_end_pos:ann_relative_end_pos+200]
                            text_snippet_full = text_paragraph[0:ann_relative_start_pos] + '*****' + text_paragraph[ann_relative_start_pos:ann_relative_end_pos] + '*****' + text_paragraph[ann_relative_end_pos:] # content ***** original mention ***** content
                            #text_snippet_full = text_snippet_full.replace("\'","") #remove single quotes; this will ensure that the stored list in row['umls_RD;doc_structure;text_snippet_full;in_text;label'] all separared by single quote only.
                            
                            #filter out the abbreviation-like matching in SemEHR - by simply using length of in-text string as a threshold
                            if len(in_text_ann) <= SemEHR_in_text_matching_len_threshold:
                                #print(in_text_ann,umls_label)
                                dict_unselected_ann_freq[umls_code + ':' + in_text_ann + ':' + umls_label] += 1
                                continue # ignore this annotation
                            
                            #dict_icd_code = mergeDict(dict_icd_code,get_code_from_url(umls_code))
                            #only consider the rare disease umls - those matched to ORDO
                            if dict_rare_disease_umls.get(umls_code,None) != None:
                                if dict_umls_code_per_row.get(umls_code,None) == None: # to avoid umls duplication in each row, but this also removes the frequency info.
                                    df.at[index,'umls_RD'].append(umls_code)
                                    #also store the document structure that the umls_code is matched to
                                    df.at[index,'doc_structure'].append(doc_part + ':' + umls_code)
                                    
                                dict_umls_code_per_row[umls_code] += 1
                                df.at[index,'umls_RD;doc_structure;text_snippet_full;in_text;label'].append(umls_code + ';' + doc_part + ';' + text_snippet_full + ';' + str(in_text_ann) + ';' + str(umls_label))
                                #df.at[index,'text_snippets'].append(text_snippet + ':' + umls_code)
                    #df.at[index,'icd9'] = dict_icd_code.keys()
                    #df.at[index,'umls'] = dict_umls_code.keys()
                    #icd_9_code_set = ";".join(dict_code.keys())
                    #outfile.write(','.join([line[0], line[1], line[2], text, line[4], prediction, str(len(tokens))]) + '\n')
                    
            else:
                #for radiology reports and other documents (e.g. ESS and Tayside), parsing the json based on the minimum installation of SemEHR https://github.com/CogStack/CogStack-SemEHR/tree/safehaven_mini/installation
                doc_part = '' # document structure name is not supported for the minimum installation version, thus as empty string
                text_snippet_full = '' # this is as empty string here for the output since there is no doc structure, the snippet is the full text and too long to be displayed here
                anns_umls = data['annotations'][0] # get the UMLS annotations, which is the first part of all 'annotations', the rest two parts are gazeteer-based phenotypes and the sentence splits
                for ann in anns_umls:
                    ann_features = ann["features"]
                        
                    # filtering based on the annotation features
                    if ann_features["Negation"] == "Affirmed" and ann_features["STY"] == "Disease or Syndrome" and ann_features["Experiencer"] == "Patient":
                        in_text_ann = ann_features["string_orig"]
                        umls_code = ann_features["inst"]
                        umls_label = ann_features["PREF"]
                        
                        #get positions
                        ann_start_ = ann['startNode']
                        ann_start_pos = ann_start_['offset']
                        ann_end_ = ann['endNode']
                        ann_end_pos = ann_end_['offset']
                        
                        #get text snippet: the part of text that contain the string mention identified from SemEHR
                        #text_snippet_full = text_full[0:ann_start_pos] + '*****' + text_paragraph[ann_start_pos:ann_end_pos] + '*****' + text_paragraph[ann_end_pos:] # content ***** original mention ***** content
                        
                        #filter out the abbreviation-like matching in SemEHR - by simply using length of in-text string as a threshold
                        if len(in_text_ann) <= SemEHR_in_text_matching_len_threshold:
                            #print(in_text_ann,umls_label)
                            dict_unselected_ann_freq[umls_code + ':' + in_text_ann + ':' + umls_label] += 1
                            continue # ignore this annotation
                        
                        #dict_icd_code = mergeDict(dict_icd_code,get_code_from_url(umls_code))
                        #only consider the rare disease umls - those matched to ORDO
                        if dict_rare_disease_umls.get(umls_code,None) != None:
                            if dict_umls_code_per_row.get(umls_code,None) == None: # to avoid umls duplication in each row, but this also removes the frequency info.
                                df.at[index,'umls_RD'].append(umls_code)
                                
                            dict_umls_code_per_row[umls_code] += 1
                            df.at[index,'umls_RD;doc_structure;text_snippet_full;in_text;label'].append(umls_code + ';' + doc_part + ';' + text_snippet_full + ';' + str(in_text_ann) + ';' + str(umls_label)) # here doc_part is '' so there will be a ';;' in the string value; also not showing the full text (i.e. set text_snippet_full as '') since it is too long
                            df.at[index,'mention_offsets'].append('%d %d' % (ann_start_pos,ann_end_pos))
                            
                                
    except json.decoder.JSONDecodeError as err:
        print(err)
        print(json_f_name)
    except FileNotFoundError as err:
        print(err)

print('unselected anns by SemEHR_in_text_matching_len_threshold:', len(dict_unselected_ann_freq), dict_unselected_ann_freq)

# (we actually did not do any filtering, this was done by setting the prevalence_percentage_threshold to 1 in the setting section)
# *filtering* out the rare disease UMLS code based on percentage in all documents, with a user-specified threshold (To Do: with tf-idf)
# based on the estimated prevalence_percentage_threshold from the user, with the df.at[index,'umls_RD'] values stored previously.
dict_umls_valid = defaultdict(int)
for i, row in df.iterrows():
    for umls_RD in row['umls_RD']:
        dict_umls_valid[umls_RD] = dict_umls_valid[umls_RD] + 1
num_doc = len(df.index)
for i, row in df.iterrows():
    umls_RD_list_filtered = []
    for umls_RD in row['umls_RD']: 
        # by estimated prevalence
        if dict_umls_valid[umls_RD]/float(num_doc) <= prevalence_percentage_threshold:
            umls_RD_list_filtered.append(umls_RD)
        else:
            #remove the document structures (associated to the UMLSs)
            doc_structure_filtered = []
            for ele in row['doc_structure']:
                if not ':' + umls_RD in ele:
                    doc_structure_filtered.append(ele)
            row['doc_structure'] = doc_structure_filtered
            
            #remove the umls_RD from 'umls_RD;doc_structure;text_snippet_full;in_text;label' as well
            umls_RD_in_text_label_list_filtered = []            
            for ele in row['umls_RD;doc_structure;text_snippet_full;in_text;label']:
                if not umls_RD + ';' in ele:
                    umls_RD_in_text_label_list_filtered.append(ele)
            row['umls_RD;doc_structure;text_snippet_full;in_text;label'] = umls_RD_in_text_label_list_filtered
            
    # update the dataframe with filtered UMLSs
    df.at[i,'doc_structure'] = row['doc_structure']
    df.at[i,'umls_RD'] = umls_RD_list_filtered # this covered the previous, non-filtered value.
    df.at[i,'umls_RD;doc_structure;text_snippet_full;in_text;label'] = row['umls_RD;doc_structure;text_snippet_full;in_text;label']    
    
# 3. map rare disease UMLS to ORDO and icd9_RD
map = pd.read_excel('./ontology/ORDO2UMLS_ICD10_ICD9+titles_final_v2.xlsx',engine='openpyxl')

df['ORDO_ID'] = ""
df['ORDO_ID'] = df['ORDO_ID'].apply(list)
df['ORDO_pref_label'] = ""
df['ORDO_pref_label'] = df['ORDO_pref_label'].apply(list)
df['icd9_RD'] = ""
df['icd9_RD'] = df['icd9_RD'].apply(list)
# df['icd9_RD_short_tit'] = ""
# df['icd9_RD_short_tit'] = df['icd9_RD_short_tit'].apply(list)
# df['icd9_RD_long_tit'] = ""
# df['icd9_RD_long_tit'] = df['icd9_RD_long_tit'].apply(list)
pattern = "'(.*?)'"
for i, row in df.iterrows():
    #print(type(row['umls_RD']),row['umls_RD'])
    #umls_RDs = re.findall(pattern,row['umls_RD'])
    dict_icd9_per_row = defaultdict(int)
    for j,k in enumerate(row['umls_RD']):
        umls_RD_tmp = 'UMLS:' + k + '\''
        matched_df=map[map['UMLS IDs'].str.contains(umls_RD_tmp)] # will this match to multiple rows? - to check, noted on 2 Mar 2021 - yes if available, but usually just one. The output matched_df can have multiple rows if multiple matches. - noted 29 Mar 2021
        ordo_ID_tmp = matched_df['ORDO ID'].to_string(index=False)
        ordo_pref_label_tmp = matched_df['Preferred Label'].to_string(index=False)
        icd9_RDs_tmp = matched_df['icd9-NZ'].to_string(index=False)
        icd9_RDs_tmp = re.findall(pattern,icd9_RDs_tmp)
        # icd9_RDs_short_tit_tmp = matched_df['icd9-short-titles'].to_string(index=False)
        # icd9_RDs_short_tit_tmp = re.findall(pattern,icd9_RDs_short_tit_tmp)
        # icd9_RDs_long_tit_tmp = matched_df['icd9-long-titles'].to_string(index=False)
        # icd9_RDs_long_tit_tmp = re.findall(pattern,icd9_RDs_long_tit_tmp)
        #if icd9_tmp != 'Series([], )':
        for icd9_RD_tmp in icd9_RDs_tmp:
            icd9_RD_tmp = icd9_RD_tmp.strip()
            if dict_icd9_per_row.get(icd9_RD_tmp,None) == None: # to avoid duplication
                df.at[i,'icd9_RD'].append(icd9_RD_tmp)
                # now map icd9 to icd9-short-title and icd9-long-titles
                #matched_df=map[map['icd9'].str.contains(umls_RD_tmp)]
            dict_icd9_per_row[icd9_RD_tmp] = dict_icd9_per_row[icd9_RD_tmp] + 1            
        df.at[i,'ORDO_ID'].append(ordo_ID_tmp) # this is a one to one output: each UMLS will output one set of ORDO ID(s) (or ordo_ID_tmp)
        df.at[i,'ORDO_pref_label'].append(ordo_pref_label_tmp) # this is a one to one output: each UMLS will output one set of ORDO ID(s)

# 4. compare manual icd9 with ontology-based rare disease icd9
df['icd9_common'] = ""
df['icd9_new_RD'] = ""
df['icd9_common_short_tit'] = ""
df['icd9_common_long_tit'] = ""
df['icd9_new_RD_short_tit'] = ""
df['icd9_new_RD_long_tit'] = ""
for i, row in df.iterrows():
    list_mimic_icd9 = row['ICD9_CODE']
    #print(list_mimic_icd9,type(list_mimic_icd9))
    list_rare_disease_icd9 = row['icd9_RD']
    df.at[i,'icd9_common'] = [code for code in list_rare_disease_icd9 if code in list_mimic_icd9]
    df.at[i,'icd9_new_RD'] = [code for code in list_rare_disease_icd9 if not code in list_mimic_icd9]
    # map to short titles and long titles
    mapped_df_list_common = [(code,map_d_icd_diagnoses[map_d_icd_diagnoses['ICD9_CODE']==code]) for code in df.at[i,'icd9_common']]
    mapped_df_list_new_RD = [(code,map_d_icd_diagnoses[map_d_icd_diagnoses['ICD9_CODE']==code]) for code in df.at[i,'icd9_new_RD']]
    #print(mapped_df_list)
    #if mapped_df_list != []:
    df.at[i,'icd9_common_short_tit'] = [code + ':' + mapped_df['SHORT_TITLE'].to_string(index=False).strip() if not 'Series([], )' in mapped_df['SHORT_TITLE'].to_string(index=False) else code + ':not found in D_ICD_DIAGNOSES' for code, mapped_df in mapped_df_list_common]
    df.at[i,'icd9_common_long_tit'] = [code + ':' + mapped_df['LONG_TITLE'].to_string(index=False).strip() if not 'Series([], )' in mapped_df['LONG_TITLE'].to_string(index=False) else code + ':not found in D_ICD_DIAGNOSES' for code, mapped_df in mapped_df_list_common]
    df.at[i,'icd9_new_RD_short_tit'] = [code + ':' + mapped_df['SHORT_TITLE'].to_string(index=False).strip() if not 'Series([], )' in mapped_df['SHORT_TITLE'].to_string(index=False) else code + ':not found in D_ICD_DIAGNOSES' for code, mapped_df in mapped_df_list_new_RD]
    df.at[i,'icd9_new_RD_long_tit'] = [code + ':' + mapped_df['LONG_TITLE'].to_string(index=False).strip() if not 'Series([], )' in mapped_df['LONG_TITLE'].to_string(index=False) else code + ':not found in D_ICD_DIAGNOSES' for code, mapped_df in mapped_df_list_new_RD]

# remove all cells which are empty lists
#df = df.mask(df.applymap(str).eq('[]'),'') # this will make the .xlsx look messy

# 5. for human validation: generate a sample of text-to-UMLS and UMLS-to-ORDO matchings from a number of documents - this is optional
if genData2valid:
    # generate num_samples_valid random examples for validation - the sampling is at the document level; the output is at the section-mention level, but all section-mention in a document is placed consecutively in the output.
    dict_num_rd={} # save the row index to dict
    random.seed(1234) # fix the random seed for reproducibility
    for x in range(num_samples_valid):
        pick = random.randint(0,num_doc-1)
        #while dict_num_rd.get(pick,None) != None: # ensure always distinct to have exact the num_samples_valid documents sampled
        #    pick = random.randint(0,num_doc-1)
        dict_num_rd[pick] = 1

    print(len(dict_num_rd),dict_num_rd)

    #df_for_validation: umls; text; in_text; UMLS with desc; ORDO with desc
    #create a new df from data https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-then-filling-it
    data2valid = []
    for i,row in df.iterrows():
        if dict_num_rd.get(i,None) != None:
            list_umls_texts = row['umls_RD;doc_structure;text_snippet_full;in_text;label']
            row_id = row['ROW_ID']
            if args.data_category != 'Discharge summary':
                text_full = row['TEXT']
                list_offsets = row['mention_offsets']
                assert len(list_offsets) == len(list_umls_texts)
            for ind, umls_texts in enumerate(list_umls_texts): # here it ignored documents which do not have any mentions matched to an ORDO-filtered UMLS concept.
                match_eles = umls_texts.split(';')
                
                umls_with_desc = match_eles[0] + ' ' + match_eles[-1]
                mention = match_eles[-2]
                doc_structure = match_eles[1]
                if args.data_category == 'Discharge summary':
                    text_snippet_full = ';'.join(match_eles[2:-2]) 
                else:
                    #mark the mention in the full text based on the offsets
                    pos_start, pos_end = list_offsets[ind].split(' ')
                    pos_start, pos_end = int(pos_start), int(pos_end)
                    text_snippet_full = text_full[:pos_start] + '*****' + text_full[pos_start:pos_end] + '*****' + text_full[pos_end:] #text_full[] # full text if the data is not disch sum
                
                umls_RD_tmp = 'UMLS:' + match_eles[0] + '\''
                matched_df=map[map['UMLS IDs'].str.contains(umls_RD_tmp)] # will this match to multiple rows? - yes it can
                ordo_ID_tmp = matched_df['ORDO ID'].to_string(index=False)
                ordo_pref_label_tmp = matched_df['Preferred Label'].to_string(index=False)
                ordo_with_desc = ordo_ID_tmp[26:] + ' ' + ordo_pref_label_tmp #get rid of the common web path part
                
                data2valid.append([row_id,doc_structure,text_snippet_full,mention,umls_with_desc,ordo_with_desc])
                
    df_for_validation = pd.DataFrame(data2valid,columns=['doc row ID','document structure','Text','mention','UMLS with desc', 'ORDO with desc'])
    df_for_validation.to_excel('for validation - %d docs - ori%s.xlsx' % (num_samples_valid,' - rad' if args.data_category == 'Radiology' else ''),index=False)
    
# 6. output the full matching dataframe, as .pik and as .xlxs, and the sampled data for validation.
#save the df to pickle
with open('df_MIMIC-III DS-Rare-Disease-ICD9-new-rows%s%s.pik' % (n_rows_selected,'-rad' if args.data_category == 'Radiology' else ''), 'wb') as data_f:
	pickle.dump(df, data_f)
#save the df to excel
#df.to_excel('MIMIC-III DS-Rare-Disease-ICD9.xlsx',index=False)
df.to_excel('MIMIC-III DS-Rare-Disease-ICD9-new%s.xlsx' % ('-rad' if args.data_category == 'Radiology' else ''),index=False)

