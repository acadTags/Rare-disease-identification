# process Tayside brain imaging reports with SemEHR outputs and link to ORDO, ICD9 for coding comparison, also generate samples for validation
# input: some settings, original full Tayside data, SemEHR processed Tayside brain imaging reports
# output: (i) dataframe (.pik) and (ii) excel sheet containing UMLS, ORDO, ICD9 codes identified from discharge summaries; (iii) sampled data for validation regarding the mention-to-UMLS matching by SemEHR and UMLS-to-ORDO matching from ORDO ontology

import pandas as pd
import os
from tqdm import tqdm
import requests
import csv
import json
from mimic3_data_processing_util import get_code_from_url,mergeDict,get_rare_disease_umls
from collections import defaultdict
import re
import random
import pickle
import sys
from constants import SemEHR_DIR_tayside_full
import argparse

# 0. settings
parser = argparse.ArgumentParser(description="step1 - process non-MIMIC (and no ICD) reports with SemEHR outputs and link to ORDO, ICD9 for coding comparison, also generate samples for validation")
parser.add_argument('-d','--dataset', type=str,
                    help="dataset name", default='Tayside')
parser.add_argument('-a','--all-disease', 
                    help="whether to get all diseases, otherwise, will only filter the UMLS to those have a link to ORDO",action='store_true', default=False)                    
args = parser.parse_args()

#set the SemEHR output json file directory path 
if args.dataset == 'Tayside':
    SemEHR_DIR_path = SemEHR_DIR_tayside_full
    # data file name
    filename = 'Tayside Scan Reports.txt'
    csv_delimiter = '|'
    header = 0 # header to be inferred
    header_list = None # no customed header list

# set percentage threshold to filter out "frequent" matched rare disease UMLSs - here we don't do filtering, but give an option.
prevalence_percentage_threshold = 1#0.005 # this should be the prevalence of rare disease in the ICU-admitted patients. 
# Prevalence of rare disease in the US is about 7.5/10000, see [1] J. Textoris and M. Leone, ‘Genetic Aspects of Uncommon Diseases’, in Uncommon Diseases in the ICU, M. Leone, C. Martin, and J.-L. Vincent, Eds. Cham: Springer International Publishing, 2014, pp. 3–11.
SemEHR_in_text_matching_len_threshold = 0#3 # to avoid wrong SemEHR UMLS matching due to meanings in abbreviations and short length

# for selecting only first k rows
n_rows_selected = None #None #1000, 5000, None

# weather to generate data for validation
genData2valid = True
# number of random doc samples for validation
num_samples_valid = 5000 #500 for discharge summaries in the study

# avoid ... in showing long sequence
pd.set_option("display.max_colwidth", 10000)

# 1. load Tayside original full data as a dataframe
df = pd.read_csv(filename,nrows=n_rows_selected, delimiter=csv_delimiter,header=header,names=header_list)
#print(df)
#df.to_csv('df-test-tayside.csv',index=False)
#sys.exit(0)

# 2. retrieve rare disease UMLS from SemEHR output

# load a list of selected rare disease UMLS from ORDO
dict_rare_disease_umls = get_rare_disease_umls()

#df['icd9'] = ""
#df['icd9'] = df['icd9'].apply(list)
#df['text_snippets'] = ""
#df['text_snippets'] = df['text_snippets'].apply(list)
df['doc_structure'] = "" # the document structure where the mention appeared
df['doc_structure'] = df['doc_structure'].apply(list)
#add offsets if the data category is radiology reports or others, fpr in disch the offsets are within a document structure.
#offsets for discharge summaries can be derived from here as well (but we didn't)

df['mention_offsets'] = ""
df['mention_offsets'] = df['mention_offsets'].apply(list)
df['mention'] = ""
df['mention'] = df['mention'].apply(list)
df['umls_RD'] = ""
df['umls_RD'] = df['umls_RD'].apply(list)
df['umls_RD;doc_structure;text_snippet_full;in_text;label'] = ""
df['umls_RD;doc_structure;text_snippet_full;in_text;label'] = df['umls_RD;doc_structure;text_snippet_full;in_text;label'].apply(list)

if not args.dataset == 'Tayside':
    df['id'] = ""
#dict to collect the filtered out umls annotations - cui_in-text_pref-label:frequency in all doc ann
dict_unselected_ann_freq = defaultdict(int)

for index, row in tqdm(df.iterrows()):
    #print(row['c1'], row['c2'])
    if args.dataset == 'Tayside':
        row_id = str(row['id'])
    else:
        row_id = row['filename_path'].split('/')[-1][:4]
        df.at[index,'id'] = row_id
    text_full = str(row['report'])
    json_f_name = '%s/%s.json' % (SemEHR_DIR_path, row_id)    
    try:
        with open(json_f_name) as json_file:
            data = json.load(json_file)
            
            #dict_icd_code={}
            dict_umls_code_per_row=defaultdict(int)
            
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
                    
                    #validate whether the in_text_ann is the same as the postions from SemEHR
                    try:
                        assert in_text_ann.lower() == text_full[ann_start_pos:ann_end_pos].lower()
                    except AssertionError:
                        print(in_text_ann, text_full[ann_start_pos:ann_end_pos])
                    
                    #filter out the abbreviation-like matching in SemEHR - by simply using length of in-text string as a threshold
                    if len(in_text_ann) <= SemEHR_in_text_matching_len_threshold:
                        #print(in_text_ann,umls_label)
                        dict_unselected_ann_freq[umls_code + ':' + in_text_ann + ':' + umls_label] += 1
                        continue # ignore this annotation
                    
                    #dict_icd_code = mergeDict(dict_icd_code,get_code_from_url(umls_code))
                    #only consider the rare disease umls - those matched to ORDO
                    if dict_rare_disease_umls.get(umls_code,None) != None or args.all_disease:
                        if dict_umls_code_per_row.get(umls_code,None) == None: # to avoid umls duplication in each row, but this also removes the frequency info.
                            df.at[index,'umls_RD'].append(umls_code)
                            
                        dict_umls_code_per_row[umls_code] += 1
                        df.at[index,'umls_RD;doc_structure;text_snippet_full;in_text;label'].append(umls_code + ';' + doc_part + ';' + text_snippet_full + ';' + str(in_text_ann) + ';' + str(umls_label).replace(';','_')) # here doc_part is '' so there will be a ';;' in the string value; also not showing the full text (i.e. set text_snippet_full as '') since it is too long. # also replace ';' in umls label to underscore
                        df.at[index,'mention_offsets'].append('%d %d' % (ann_start_pos,ann_end_pos))
                        df.at[index,'mention'].append(text_full[ann_start_pos:ann_end_pos])
                            
                                
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
            row_id = str(row['id'])
            # if args.dataset == 'Tayside':
                # row_id = str(row['id'])
            # else:
                # row_id = row['filename_path'].split('/')[-1][:4]
            text_full = row['report']
            list_offsets = row['mention_offsets']
            assert len(list_offsets) == len(list_umls_texts)
            for ind, umls_texts in enumerate(list_umls_texts): # here it ignored documents which do not have any mentions matched to an ORDO-filtered UMLS concept.
                match_eles = umls_texts.split(';')
                
                umls_with_desc = match_eles[0] + ' ' + match_eles[-1]
                mention = match_eles[-2]
                doc_structure = match_eles[1]
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
    df_for_validation.to_excel('for validation - %d docs - ori - %s - rad.xlsx' % (num_samples_valid, args.dataset.lower()),index=False)
    
# 6. output the full matching dataframe, as .pik and as .xlxs, and the sampled data for validation.
#save the df to pickle
with open('df_%s-rad-Rare-Disease-rows%s.pik' % (args.dataset,n_rows_selected), 'wb') as data_f:
	pickle.dump(df, data_f)
#save the df to excel
#df.to_csv('Tayside-rad-Rare-Disease.csv',index=False)
df.to_excel('%s-rad-Rare-Disease.xlsx' % args.dataset,index=False,engine='xlsxwriter')
