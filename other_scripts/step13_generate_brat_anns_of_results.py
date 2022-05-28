# for visualisation using brat
# generate brat annotation format files of all MIMIC-III documents regarding rare diseases
'''
reads mention-level pred results all docs[ sup].xlsx to get the mention offsets and ORDO IDs results from NLP
      MIMIC-III DS-Rare-Disease-ICD9-new-filtered[-sup]-bp.xlsx (an output of step9)
      MIMIC-III DS-Rare-Disease-ICD9-new-filtered[-sup]-NZ.xlsx (an output of step9)
      MIMIC-III DS-Rare-Disease-ICD9-new-filtered[-sup]-both.xlsx (an output of step9) to get the ORDO IDs results (in column 'ORDO_ID_icd9_manual') bassed on the manual ICD-9 codes  
      MIMIC-III DS-Rare-Disease-ICD9-new.xlsx (an output of step0) to get the full texts of discharge summaries
outputs .txt and .ann files in brat format and annotation.conf file
      
'''

import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm
import os
from rare_disease_id_util import get_ORDO_pref_label_with_dict

# avoid ... in showing long sequence
pd.set_option("display.max_colwidth", 10000)

#output str content to a file
#input:  filename and the content (str)
def output_to_file(file_name,str):
    with open(file_name, 'w', newline='\n', encoding="utf-8") as f_output: # newline as \n ensure that the output has \n for newline instead of \r\n #utf-8 as encoding to avoid BOM sign which is not supported by brat
        f_output.write(str)

def shift_offset_in_brat_ann(ann_str,shift_by_num):
    list_ann_lines = ann_str.split('\n')
    for i, ann_line in enumerate(list_ann_lines):
        #print(ann_line)
        if ann_line[0] == 'T':
            #print('went through')
            #get the offset part
            ann_line_split = ann_line.split('\t')
            concept_and_offset_str_formatted_tmp = ann_line_split[1]
            concept_pos_ind = concept_and_offset_str_formatted_tmp.find(' ')
            offset_str_formatted_tmp = concept_and_offset_str_formatted_tmp[concept_pos_ind+1:]
            #update the offset part
            offset_str_formatted_updated_tmp = shift_offset_in_formatted_str(offset_str_formatted_tmp,shift_by_num)
            #update the line of ann
            ann_line_split[1] = concept_and_offset_str_formatted_tmp[:concept_pos_ind] + ' ' + offset_str_formatted_updated_tmp
            list_ann_lines[i] = '\t'.join(ann_line_split)
    return '\n'.join(list_ann_lines)        
    
def shift_offset_in_formatted_str(offset_str_formatted,shift_by_num):
    list_offset_str_groups = offset_str_formatted.split(';')
    list_offset_str_groups_updated = []
    for offset_str_group in list_offset_str_groups:
        offsets = offset_str_group.split(' ')
        offsets_updated = [str(int(offset) + shift_by_num) for offset in offsets]
        list_offset_str_groups_updated.append(' '.join(offsets_updated))
    return ';'.join(list_offset_str_groups_updated)
    
if __name__ == '__main__':
    
    exact_or_narrower_only = True # whether using exact or narrower only matching from ORDO to ICD-10 (if setting to True, this will result in a set of rare disease ICD-9 codes with higher precision but lower recall. This will affect sources of 'NZ'.)
    
    brat_file_ann_folder_name = 'doc_rd_pred%s' % '_NZ_E_N' if exact_or_narrower_only else ''
    # create the output folder if not existed
    if not os.path.exists('./' + brat_file_ann_folder_name):
        os.makedirs('./' + brat_file_ann_folder_name)
    
    NLP_results_only = False # whether to only generate brat files for the document with at least one NLP-identified mentions. If choosing False, then all the document will be generated.
    
    show_new_in_ICD = False # whether to show 'new_in_ICD' entity tag in brat ann files, set as False by default, if True, this may result in too many annotatons and thus cause 'Error: ActiongetCollectionInformation failed on error Gateway Time-out', see https://github.com/nlplab/brat/issues/1143
    
    # read the results based on manual icd codes 
    print('reading results based on manual icd codes: df_icd files')
    #df_icd_both = pd.read_excel('./MIMIC-III DS-Rare-Disease-ICD9-new-filtered-sup-both.xlsx')
    df_icd_bp = pd.read_excel('./MIMIC-III DS-Rare-Disease-ICD9-new-filtered-sup-bp.xlsx')
    if exact_or_narrower_only: 
        # icd from NZ (ORDO-ICD10-ICD9) with exact or narrower matching from ORDO to ICD10
        df_icd_NZ = pd.read_excel('./MIMIC-III DS-Rare-Disease-ICD9-new-filtered-sup-NZ-E-N.xlsx')
    else: 
        # no filtering in icd from NZ (i.e. also with broader matching)
        df_icd_NZ = pd.read_excel('./MIMIC-III DS-Rare-Disease-ICD9-new-filtered-sup-NZ.xlsx')
        
    # read the results based on model predictions
    print('reading results based on model predictions')
    nrows=None # None for all rows or 200 for testing with a smaller set    
    df = pd.read_excel('./mention-level pred results all docs sup.xlsx',nrows=nrows) # strongly supervised
    df_weak = pd.read_excel('./mention-level pred results all docs.xlsx',nrows=nrows) # weak supervised    
    
    #create sub dataframe
    sub_df = df[["doc row ID", "document structure","Text","mention", "UMLS code","UMLS desc","ORDO ID list","ORDO pref label list","pred text-to-UMLS","pred UMLS-to-ORDO","pred text-to-ORDO"]]
    sub_df_weak = df_weak[["doc row ID","pred text-to-ORDO"]] #only the final pred result is needed as the other columns are mostly in sub_df
    print(type(sub_df))

    #link the SUBJECT_ID and HADM_ID of the document, through merging.
    #sub_df['doc row ID'] = sub_df['doc row ID'].astype(int)
    print(sub_df['doc row ID'])

    sub_df = sub_df.rename(columns={'doc row ID': 'ROW_ID'})
    sub_df_weak = sub_df_weak.rename(columns={'doc row ID': 'ROW_ID'})

    df_all_docs = pd.read_excel('MIMIC-III DS-Rare-Disease-ICD9-new.xlsx',sheet_name="Sheet1") # this also contains the full texts of all docs

    sub_df_expanded = pd.merge(sub_df, df_all_docs, how="right", on=['ROW_ID'])
    sub_df_weak_expanded = pd.merge(sub_df_weak, df_all_docs, how="right", on=['ROW_ID'])
    print(len(sub_df_expanded))
    sub_df_expanded_sel = sub_df_expanded[["ROW_ID", "SUBJECT_ID", "HADM_ID", "TEXT", "document structure","Text","mention", "UMLS code","UMLS desc","ORDO ID list","ORDO pref label list","pred text-to-UMLS","pred UMLS-to-ORDO","pred text-to-ORDO"]]
    sub_df_expanded_sel = sub_df_expanded_sel.rename(columns={'document structure': 'document structure name', 'TEXT': 'TEXT_whole_doc', 'Text': 'TEXT_in_doc_structure'})
    #sub_df_expanded_sel = sub_df_expanded_sel[sub_df_expanded_sel["pred text-to-ORDO"]==1] # only consider those predicted as True
    
    num_offset_errs=0
    #loop over the df to get the document level offsets
    for i, row in tqdm(sub_df_expanded_sel.iterrows()):
        #infer document-level offsets
        text_whole_doc = row['TEXT_whole_doc']
        text_in_doc_struc = row['TEXT_in_doc_structure']
        if pd.isna(text_in_doc_struc): # not processing the rows when there is no mentions identified by SemEHR (thus text_in_doc_struc is nan after the merging)
            continue
        #text_in_doc_struc = '' if pd.isna(text_in_doc_struc) else text_in_doc_struc # if is nan, then replace it to empty string.
        #print(i, text_in_doc_struc)
        #mention = row['mention']
        #offset of mention in doc struc
        match = re.search("\*{5,5}(.*?)\*{5,5}", text_in_doc_struc, re.DOTALL)
        ind_start_men_in_ds = match.start()
        ind_end_men_in_ds = match.end()
        #offset of doc struc in doc 
        text_in_doc_struc_non_markup = re.sub("\*{5,5}", "", text_in_doc_struc, re.DOTALL)
        ind_start_ds_in_doc = text_whole_doc.find(text_in_doc_struc_non_markup) # 
        #offset of mention in doc - inferred from the above two offsets
        ind_start_men_in_doc = ind_start_men_in_ds + ind_start_ds_in_doc
        ind_end_men_in_doc = (ind_end_men_in_ds - 10) + ind_start_ds_in_doc # minus one, since the end offset was right shifted twice                        
        #validate the doc-level offset and if not correct, find the second doc struc offset and update the doc-level offset
        mention_in_text = text_whole_doc[ind_start_men_in_doc:ind_end_men_in_doc] # or match.group(0)
        while mention_in_text != match.group(0)[5:-5] and ind_start_ds_in_doc != -1: # keep searching the doc structure and validating until not found
            ind_start_ds_in_doc = text_whole_doc.find(text_in_doc_struc_non_markup,ind_start_ds_in_doc+len(text_in_doc_struc_non_markup)) 
            #offset of mention in doc - inferred from the above two offsets
            ind_start_men_in_doc = ind_start_men_in_ds + ind_start_ds_in_doc
            ind_end_men_in_doc = (ind_end_men_in_ds - 10) + ind_start_ds_in_doc # minus one, since the end offset was right shifted twice
            mention_in_text = text_whole_doc[ind_start_men_in_doc:ind_end_men_in_doc] # or match.group(0)
            
        try:
            assert mention_in_text == match.group(0)[5:-5], print('assertion error data at',i,'mention_in_text:',mention_in_text,'\nmatch.group(0):',match.group(0)[5:-5]) # check whether the full text offsets are correct
        except AssertionError as e:            
            num_offset_errs = num_offset_errs + 1
            
        # get offsets (of different parts) formatted as in the .ann file, parts separated by '\n', usually just 1 part
        # support discontinous annotations in brat v1.3         
        mention_in_text_parts = mention_in_text.split('\n')
        for ind_men, mention_in_text_part in enumerate(mention_in_text_parts):
            if ind_men == 0:
                ind_end_men_in_doc_interium = ind_start_men_in_doc + len(mention_in_text_part)
                offset_str_formatted = '%s %s' % (ind_start_men_in_doc,ind_end_men_in_doc_interium)
            else:
                ind_start_men_in_doc_interium = ind_end_men_in_doc_interium + 1
                ind_end_men_in_doc_interium = ind_start_men_in_doc_interium + len(mention_in_text_part)
                offset_str_formatted = offset_str_formatted + ';' + '%s %s' % (ind_start_men_in_doc_interium,ind_end_men_in_doc_interium) 
                #here pos_end is overriding the prevoius value, this is for getting the final pos_end.

        sub_df_expanded_sel.at[i,'mention orig'] = ' '.join(mention_in_text_parts) # this is the mention in the text where each newline is replaced by a whitespace
        sub_df_expanded_sel.at[i,'mention offset in full document'] = offset_str_formatted#'%s %s' % (str(ind_start_men_in_doc),str(ind_end_men_in_doc))
        #print(sub_df_expanded_sel.at[i,'mention orig'],sub_df_expanded_sel.at[i,'mention offset in full document'])
    #sub_df_expanded_sel.to_csv('full_set_RD_ann_pred_MIMIC_III_disch_sup.csv',index=False)
    #print(sub_df_expanded_sel)
    print('number of offset errors:', num_offset_errs)  # 766 errors, error rate 0.6% and this is mostly related to the SemEHR that it can recognise mention when there is newline in the mention.
    
    # loop over the updated df again to generate document-level .txt and .ann files
    list_source_type = ['NLP-strong','NLP-weak','ICD-bp','ICD-NZ']
    
    dict_ORDO_formatted = defaultdict(int)
    dict_ORDO_in_each_doc_strong = defaultdict(int)
    dict_ORDO_in_each_doc_weak = defaultdict(int)
    dict_ORDO_to_pref_label = {}
    map=None
    pattern = "'(.*?)'"
    prev_doc_row_ID = ''
    #text_whole_doc=''
    filename = ''
    #cohort doc csv file - as a summary of all rare disease cohorts
    cohort_doc = 'doc-subject_ID-row_ID-rORDO_IDs_NLP_strong,ORDO ID by NLP strong,ORDO ID by NLP weak,ORDO ID by ICD-bp,ORDO ID by ICD-NZ' + '\n'
    for i, row in tqdm(sub_df_expanded_sel.iterrows()):
        pred_strong = row['pred text-to-ORDO']
        pred_weak = sub_df_weak_expanded.at[i,'pred text-to-ORDO']
        
        if NLP_results_only:
            if (pred_strong != 1) and (pred_weak != 1): # only keep those having a positive pred in either strong or weak models
                continue
        
        doc_row_ID = row['ROW_ID']
        
        #get ORDO ID with desc and save to dictionary
        ORDO_ID_list_str = row['ORDO ID list']
        ORDO_ID_list_str = '[]' if pd.isna(ORDO_ID_list_str) else ORDO_ID_list_str
        ORDO_ID_list = re.findall(pattern,ORDO_ID_list_str)
        ORDO_pref_label_list_str = row['ORDO pref label list']
        ORDO_pref_label_list_str = '[]' if pd.isna(ORDO_pref_label_list_str) else ORDO_pref_label_list_str        
        ORDO_pref_label_list = re.findall(pattern,ORDO_pref_label_list_str)
        
        ORDO_ID_with_desc_formatted_list = []
        for ORDO_ID, ORDO_pref_label in zip(ORDO_ID_list, ORDO_pref_label_list):
            ORDO_ID_with_desc = ORDO_ID + ' ' + ORDO_pref_label
            ORDO_ID_with_desc_formatted = re.sub('[^a-zA-Z0-9_-]+','_',ORDO_ID_with_desc)
            ORDO_ID_with_desc_formatted_list.append(ORDO_ID_with_desc_formatted)
            dict_ORDO_formatted[ORDO_ID_with_desc_formatted] += 1
        
        offset_str_formatted = row['mention offset in full document']
        offset_str_formatted = '' if pd.isna(offset_str_formatted) else offset_str_formatted
        #offsets = offset_str_formatted.split()
        #prev_text_whole_doc = text_whole_doc
        #text_whole_doc = row['TEXT_whole_doc']            
        #mention_in_text = text_whole_doc[int(offsets[0]):int(offsets[1])]        
        mention_orig = row['mention orig']
        mention_orig = '' if pd.isna(mention_orig) else mention_orig
        #print(offset_str_formatted,mention_orig)

        if doc_row_ID != prev_doc_row_ID:
            #if row_ID appeared for the first time, 
            #(i)save the prev .ann file and .txt file (if the prev filename is not empty)
            #(ii)extract information for the annotation of the current new doc
            #***for the prev doc***
            if filename != '':
                #get the list of predicted ORDO IDs by strong or weak NLP models for the document (and clear the dict)
                ORDO_ID_with_desc_list_NLP_strong = list(dict_ORDO_in_each_doc_strong.keys())
                dict_ORDO_in_each_doc_strong.clear()
                ORDO_ID_with_desc_list_NLP_weak = list(dict_ORDO_in_each_doc_weak.keys())
                dict_ORDO_in_each_doc_weak.clear()
                
                #update filename with ORDO IDs
                ORDO_ID_list_NLP_strong = [ORDO_ID_with_desc_strong.split('_')[1] for ORDO_ID_with_desc_strong in ORDO_ID_with_desc_list_NLP_strong]
                filename = filename + '-r' + '-r'.join(ORDO_ID_list_NLP_strong) if len(ORDO_ID_list_NLP_strong)>0 else filename # each ORDO ID prefixed with an 'r'.
                
                #update and save .txt document
                #create summary and add it to the beginning of the whole document
                summary_pre_doc = 'ORDO ID by NLP strong: %s\nORDO ID by NLP weak: %s\nORDO ID by ICD-bp: %s\nORDO ID by ICD-NZ: %s' % (' '.join(ORDO_ID_with_desc_list_NLP_strong), ' '.join(ORDO_ID_with_desc_list_NLP_weak), ' '.join(ORDO_ID_with_desc_list_manual_ICD_bp), ' '.join(ORDO_ID_with_desc_list_manual_ICD_NZ)) + '\n\n' #double newlines to make it a seperate paragraph
                text_whole_doc_with_sum = summary_pre_doc + text_whole_doc                
                output_to_file('./' + brat_file_ann_folder_name + '/' + filename + '.txt',text_whole_doc_with_sum)
                #update the cohort doc
                summary_pre_doc_for_cohort_csv = '%s,%s,%s,%s' % (' '.join(ORDO_ID_with_desc_list_NLP_strong), ' '.join(ORDO_ID_with_desc_list_NLP_weak), ' '.join(ORDO_ID_with_desc_list_manual_ICD_bp), ' '.join(ORDO_ID_with_desc_list_manual_ICD_NZ))
                cohort_doc = cohort_doc + filename + ',' + summary_pre_doc_for_cohort_csv + '\n'
                
                if has_ann:
                    #update annotation offsets and save .ann file
                    annotation_updated = shift_offset_in_brat_ann(annotation, len(summary_pre_doc))
                    #print('let\'s compare')
                    #print(annotation)
                    #print(annotation_updated)
                    
                    #add marking for the summary_pre_doc section - what's new in NLP or ICD compared to the other?
                    ORDO_ID_set_NLP = set(ORDO_ID_with_desc_list_NLP_strong + ORDO_ID_with_desc_list_NLP_weak)
                    ORDO_ID_set_ICD = set(ORDO_ID_with_desc_list_manual_ICD_bp + ORDO_ID_with_desc_list_manual_ICD_NZ)
                    ORDO_ID_set_new_in_NLP = ORDO_ID_set_NLP - ORDO_ID_set_ICD
                    ORDO_ID_set_new_in_ICD = ORDO_ID_set_ICD - ORDO_ID_set_NLP
                    #get the current ann index
                    ann_lines = annotation_updated.split('\n')
                    ann_lines.reverse()
                    ind_ann=0
                    for ann_line in ann_lines:
                        if ann_line[0] == 'T':
                            ind_ann = int(ann_line[1:ann_line.find('\t')])
                            break
                    #update annotations
                    list_new_sets = [ORDO_ID_set_new_in_NLP,ORDO_ID_set_new_in_ICD] if show_new_in_ICD else [ORDO_ID_set_new_in_NLP]
                    list_new_marks = ['new-in-NLP', 'new-in-ICD'] if show_new_in_ICD else ['new-in-NLP'] # for marking what's new in NLP or ICD in the summary section
                    for ORDO_ID_new_set,new_mark in zip(list_new_sets,list_new_marks):
                        for ORDO_ID in ORDO_ID_new_set:
                            offset_str_formatted_new_list = ['%d %d' % (m.start(),m.end()) for m in re.finditer(ORDO_ID, summary_pre_doc)]
                            for offset_str_formatted_new in offset_str_formatted_new_list:
                                ind_ann = ind_ann + 1                    
                                annotation_updated = annotation_updated + '\n' + 'T%d	%s %s	%s' % (ind_ann,new_mark, offset_str_formatted_new, ORDO_ID)
                    
                    output_to_file('./' + brat_file_ann_folder_name + '/' + filename + '.ann', annotation_updated)
            
            #***for the current/new doc***
            text_whole_doc = row['TEXT_whole_doc']                        
            
            #retrieve ORDO_IDs (and their descriptions) based on the manual ICD codes - two sources, bp and NZ
            df_icd_bp_retrieved = df_icd_bp[df_icd_bp['ROW_ID']==doc_row_ID]
            assert len(df_icd_bp_retrieved) == 1
            ORDO_ID_list_manual_ICD_bp_str = df_icd_bp_retrieved['ORDO_ID_icd9_manual'].to_string(index=False)
            ORDO_ID_list_manual_ICD_bp = re.findall(pattern,ORDO_ID_list_manual_ICD_bp_str)
            ORDO_ID_with_desc_list_manual_ICD_bp = []
            for ORDO_ID in ORDO_ID_list_manual_ICD_bp:
                ORDO_pref_label, dict_ORDO_to_pref_label,map = get_ORDO_pref_label_with_dict(ORDO_ID,dict_ORDO_to_pref_label,map=map)
                ORDO_ID_with_desc = ORDO_ID + ' ' + ORDO_pref_label
                ORDO_ID_with_desc_formatted = re.sub('[^a-zA-Z0-9_-]+','_',ORDO_ID_with_desc)
                ORDO_ID_with_desc_list_manual_ICD_bp.append(ORDO_ID_with_desc_formatted)
            df_icd_NZ_retrieved = df_icd_NZ[df_icd_NZ['ROW_ID']==doc_row_ID]
            assert len(df_icd_NZ_retrieved) == 1
            ORDO_ID_list_manual_ICD_NZ_str = df_icd_NZ_retrieved['ORDO_ID_icd9_manual'].to_string(index=False)
            ORDO_ID_list_manual_ICD_NZ = re.findall(pattern,ORDO_ID_list_manual_ICD_NZ_str)
            ORDO_ID_with_desc_list_manual_ICD_NZ = []
            for ORDO_ID in ORDO_ID_list_manual_ICD_NZ:
                ORDO_pref_label, dict_ORDO_to_pref_label, map = get_ORDO_pref_label_with_dict(ORDO_ID,dict_ORDO_to_pref_label,map=map)
                #print('ORDO_pref_label',ORDO_pref_label)
                ORDO_ID_with_desc = ORDO_ID + ' ' + ORDO_pref_label
                ORDO_ID_with_desc_formatted = re.sub('[^a-zA-Z0-9_-]+','_',ORDO_ID_with_desc)
                ORDO_ID_with_desc_list_manual_ICD_NZ.append(ORDO_ID_with_desc_formatted)
                
            # filename for the new .txt and .ann
            doc_subject_ID = row['SUBJECT_ID']
            filename = 'doc-%s-%s' % (doc_subject_ID,doc_row_ID)
            
            has_ann = True if len(ORDO_ID_with_desc_formatted_list) > 0 else False
                
            ind_ann = 0
            attr_ann = 0
            for ind, ORDO_ID_with_desc_formatted in enumerate(ORDO_ID_with_desc_formatted_list):
                #ORDO_ID = dict_ORDO_formatted[ORDO_ID_with_desc_formatted]
                from_ICD_bp = True if ORDO_ID_with_desc_formatted in ORDO_ID_with_desc_list_manual_ICD_bp else False
                from_ICD_NZ = True if ORDO_ID_with_desc_formatted in ORDO_ID_with_desc_list_manual_ICD_NZ else False
                #print(filename)
                # if filename == 'doc-48672-4056':
                    # print('debugging here first appearance')
                    # print('doc_subject_ID:',doc_subject_ID,'doc_row_ID:',doc_row_ID)
                    # print('ORDO_ID_with_desc_formatted:',ORDO_ID_with_desc_formatted)
                    # print('ORDO_ID_with_desc_list_manual_ICD_bp:',ORDO_ID_with_desc_list_manual_ICD_bp)
                    # print('ORDO_ID_with_desc_list_manual_ICD_NZ:',ORDO_ID_with_desc_list_manual_ICD_NZ)
                #add concept type
                ind_ann = ind_ann + 1
                if ind == 0:
                    #to add weak label - 
                    #annotation = 'T%d	%s %s	%s\nA%d	NLP-strong T%d' % (ind_ann,ORDO_ID_with_desc_formatted, offset_str_formatted, mention_orig,attr_ann,ind_ann)     
                    annotation = 'T%d	%s %s	%s' % (ind_ann,ORDO_ID_with_desc_formatted, offset_str_formatted, mention_orig)                         
                else:
                    annotation = annotation + '\n' + 'T%d	%s %s	%s' % (ind_ann,ORDO_ID_with_desc_formatted, offset_str_formatted, mention_orig)
                
                list_source_criteria_bool = [pred_strong, pred_weak, from_ICD_bp, from_ICD_NZ]
                for source_criteria_bool, source_label in zip(list_source_criteria_bool, list_source_type):                    
                    if source_criteria_bool:
                        #add new attribute to the main annotation                   
                        attr_ann = attr_ann + 1
                        annotation = annotation + '\nA%d	%s T%d' % (attr_ann,source_label,ind_ann)     
                        #save the unique ORDO_IDs for strong and weak NLP models
                        if source_label == 'NLP-strong':
                            dict_ORDO_in_each_doc_strong[ORDO_ID_with_desc_formatted] += 1
                        if source_label == 'NLP-weak':
                            dict_ORDO_in_each_doc_weak[ORDO_ID_with_desc_formatted] += 1
                        
                for source_criteria_bool, source_label in zip(list_source_criteria_bool, list_source_type):
                    if source_criteria_bool:
                        #add new annotation
                        ind_ann = ind_ann + 1
                        annotation = annotation + '\n' + 'T%d	%s %s	%s' % (ind_ann,source_label, offset_str_formatted, mention_orig)
        else:
            #doc_subject_ID = row['SUBJECT_ID']
            # continuing the same doc
            for ORDO_ID_with_desc_formatted in ORDO_ID_with_desc_formatted_list:
                from_ICD_bp = True if ORDO_ID_with_desc_formatted in ORDO_ID_with_desc_list_manual_ICD_bp else False
                from_ICD_NZ = True if ORDO_ID_with_desc_formatted in ORDO_ID_with_desc_list_manual_ICD_NZ else False
                #print(filename)
                # if filename == 'doc-48672-4056':
                    # print('debugging here continuing')
                    # print('doc_subject_ID:',doc_subject_ID,'doc_row_ID:',doc_row_ID)
                    # print('ORDO_ID_with_desc_formatted:',ORDO_ID_with_desc_formatted)
                    # print('ORDO_ID_with_desc_list_manual_ICD_bp:',ORDO_ID_with_desc_list_manual_ICD_bp)
                    # print('ORDO_ID_with_desc_list_manual_ICD_NZ:',ORDO_ID_with_desc_list_manual_ICD_NZ)
                #add concept type
                ind_ann = ind_ann + 1
                #attr_ann = attr_ann + 1
                annotation = annotation + '\n' + 'T%d	%s %s	%s' % (ind_ann,ORDO_ID_with_desc_formatted, offset_str_formatted, mention_orig)
                
                list_source_criteria_bool = [pred_strong, pred_weak, from_ICD_bp, from_ICD_NZ]
                for source_criteria_bool, source_label in zip(list_source_criteria_bool, list_source_type):                    
                    if source_criteria_bool:
                        #add new attribute to the main annotation                   
                        attr_ann = attr_ann + 1
                        annotation = annotation + '\nA%d	%s T%d' % (attr_ann,source_label,ind_ann)     
                        #save the unique ORDO_IDs for strong and weak NLP models    
                        if source_label == 'NLP-strong':
                            dict_ORDO_in_each_doc_strong[ORDO_ID_with_desc_formatted] += 1
                        if source_label == 'NLP-weak':
                            dict_ORDO_in_each_doc_weak[ORDO_ID_with_desc_formatted] += 1
                        
                for source_criteria_bool, source_label in zip(list_source_criteria_bool, list_source_type):
                    if source_criteria_bool:
                        #add new annotation
                        ind_ann = ind_ann + 1
                        annotation = annotation + '\n' + 'T%d	%s %s	%s' % (ind_ann,source_label, offset_str_formatted, mention_orig)
                    
        prev_doc_row_ID = doc_row_ID
    
    #update content and save the last .ann and .txt files - the two 'output_to_file' code below is same as in the *if filename != '':* block.
    #get the list of predicted ORDO IDs by strong or weak NLP models for the document (and clear the dict)
    ORDO_ID_with_desc_list_NLP_strong = list(dict_ORDO_in_each_doc_strong.keys())
    dict_ORDO_in_each_doc_strong.clear()
    ORDO_ID_with_desc_list_NLP_weak = list(dict_ORDO_in_each_doc_weak.keys())
    dict_ORDO_in_each_doc_weak.clear()
    
    #update filename with ORDO IDs
    ORDO_ID_list_NLP_strong = [ORDO_ID_with_desc_strong.split('_')[1] for ORDO_ID_with_desc_strong in ORDO_ID_with_desc_list_NLP_strong]
    filename = filename + '-r' + '-r'.join(ORDO_ID_list_NLP_strong) if len(ORDO_ID_list_NLP_strong)>0 else filename # each ORDO ID prefixed with an 'r'.
    
    #update and save .txt document
    #create summary and add it to the beginning of the whole document
    summary_pre_doc = 'ORDO ID by NLP strong: %s\nORDO ID by NLP weak: %s\nORDO ID by ICD-bp: %s\nORDO ID by ICD-NZ:%s' % (' '.join(ORDO_ID_with_desc_list_NLP_strong), ' '.join(ORDO_ID_with_desc_list_NLP_weak), ' '.join(ORDO_ID_with_desc_list_manual_ICD_bp), ' '.join(ORDO_ID_with_desc_list_manual_ICD_NZ)) + '\n\n' #double newlines to make it a seperate paragraph
    text_whole_doc_with_sum = summary_pre_doc + text_whole_doc                
    output_to_file('./' + brat_file_ann_folder_name + '/' + filename + '.txt',text_whole_doc_with_sum)
    #update the cohort doc
    summary_pre_doc_for_cohort_csv = '%s,%s,%s,%s' % (' '.join(ORDO_ID_with_desc_list_NLP_strong), ' '.join(ORDO_ID_with_desc_list_NLP_weak), ' '.join(ORDO_ID_with_desc_list_manual_ICD_bp), ' '.join(ORDO_ID_with_desc_list_manual_ICD_NZ))
    cohort_doc = cohort_doc + filename + ',' + summary_pre_doc_for_cohort_csv + '\n'
    
    #update annotation offsets and save .ann file
    annotation_updated = shift_offset_in_brat_ann(annotation, len(summary_pre_doc))
    #print('let\'s compare')
    #print(annotation)
    #print(annotation_updated)
    
    #add marking for the summary_pre_doc section - what's new in NLP or ICD compared to the other?
    ORDO_ID_set_NLP = set(ORDO_ID_with_desc_list_NLP_strong + ORDO_ID_with_desc_list_NLP_weak)
    ORDO_ID_set_ICD = set(ORDO_ID_with_desc_list_manual_ICD_bp + ORDO_ID_with_desc_list_manual_ICD_NZ)
    ORDO_ID_set_new_in_NLP = ORDO_ID_set_NLP - ORDO_ID_set_ICD
    ORDO_ID_set_new_in_ICD = ORDO_ID_set_ICD - ORDO_ID_set_NLP
    #get the current ann index
    ann_lines = annotation_updated.split('\n')
    ann_lines.reverse()
    ind_ann=0
    for ann_line in ann_lines:
        if ann_line[0] == 'T':
            ind_ann = int(ann_line[1:ann_line.find('\t')])
            break
    #update annotations
    list_new_sets = [ORDO_ID_set_new_in_NLP,ORDO_ID_set_new_in_ICD] if show_new_in_ICD else [ORDO_ID_set_new_in_NLP]
    list_new_marks = ['new-in-NLP', 'new-in-ICD'] if show_new_in_ICD else ['new-in-NLP'] # for marking what's new in NLP or ICD in the summary section
    for ORDO_ID_new_set,new_mark in zip(list_new_sets,list_new_marks):
        for ORDO_ID in ORDO_ID_new_set:
            offset_str_formatted_new_list = ['%d %d' % (m.start(),m.end()) for m in re.finditer(ORDO_ID, summary_pre_doc)]
            for offset_str_formatted_new in offset_str_formatted_new_list:
                ind_ann = ind_ann + 1                    
                annotation_updated = annotation_updated + '\n' + 'T%d	%s %s	%s' % (ind_ann,new_mark, offset_str_formatted_new, ORDO_ID)
    
    output_to_file('./' + brat_file_ann_folder_name + '/' + filename + '.ann', annotation_updated)
    
    # output cohort doc
    output_to_file('cohort_doc%s.csv' % '_NZ_E_N' if exact_or_narrower_only else '', cohort_doc)
    
    # generate annotation.conf    
    list_source_type_for_att = [ele + '    Arg:<ENTITY>' for ele in list_source_type]
    conf_content = '''[entities]
%s

[events]

[relations]
<OVERLAP>	Arg1:<ENTITY>, Arg2:<ENTITY>, <OVL-TYPE>:<ANY>

[attributes]
%s
''' % ('\n'.join(list_new_marks + list_source_type + list(dict_ORDO_formatted.keys())),'\n'.join(list_source_type_for_att))
    output_to_file('./' + brat_file_ann_folder_name + '/' + 'annotation.conf',conf_content)
    
    # generate visual.conf
    list_source_type_colour = ['#b7f5f7','#f7f7cb','#eaffaa','#ffccaa','#aab1ff','#ffaacb'] if show_new_in_ICD else ['#b7f5f7','#eaffaa','#ffccaa','#aab1ff','#ffaacb'] # removed the second colour if now show 'new-in-ICD'
    list_source_type_with_colour = [st + '	bgColor:' + colour for st,colour in zip(list_new_marks + list_source_type,list_source_type_colour)]
    visual_content = '''[labels]
    
[drawing]	 
SPAN_DEFAULT	fgColor:black, bgColor:lightgreen, borderColor:darken
ARC_DEFAULT	color:black, dashArray:-, arrowHead:triangle-5
%s''' % '\n'.join(list_source_type_with_colour)
    output_to_file('./' + brat_file_ann_folder_name + '/' + 'visual.conf',visual_content)