# this program calculates 
# (i) micro-level and macro-level precision, recall, and F1
# (ii) instance-based precision, recall, and F1 - optional, this has issue to deal with non-rare disease documents, i.e. for this (pred-true) pairs (0,0,0 - 0,0,0) and (1,0,0 - 0，0，0） will have the same precision, recall, and F1 score, all as 0.

# this is based on the assumption that the SemEHR output + text-UMLS + UMLS-ORDO can find the complete annotation of rare diseases, this is by no means correct, but it allows us to use multi-label classification metrics for evaluation.
# input: full results sheet 'for validation - SemEHR ori.xlsx'
# output： an excel file for each model of per-label results and full results.
#         a png image file of label distribution plot

# update: to add ICD-based results
# input: (i) using the 'doc row ID' column in the mention-level results sheet 'for validation - SemEHR ori.xlsx'
#        (ii) using the 'ORDO_ID_icd9_manual' (and 'ORDO_ID_filtered') column in the admission-level pred results sheet sheet 'MIMIC-III DS-Rare-Disease-ICD9-new-filtered....xlsx'
# output： an excel file for ICD-based (and NLP+ICD-based) per-label results and full results.

# run with python -W ignore step8.1_patient_lvl_results_from_anns_rad.py if there are many warnings (UndefinedMetricWarning and RuntimeWarning).

from step8_patient_level_results_from_annotations import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
import re
# for plotting label distribution
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from adjustText import adjust_text
from tqdm import tqdm

# avoid ... in showing long sequence
pd.set_option("display.max_colwidth", 10000) # this is necessary for retrieving long str from dataframe!

if __name__ == '__main__':
    # 0. display and output setting
    dataset = 'MIMIC-III' # 'MIMIC-III' or 'Tayside'
    data_category = 'Radiology'
    display_per_label_results = False
    # output the results to excel - each method to a sheet
    output_xlsx = True
    
    # 1. load binary prediction results from sheet
    print('read validation mention-level data sheet')
    if dataset == 'MIMIC-III':
        #data_sheet_fn = 'for validation - 1000 docs - ori - MIMIC-III-rad.xlsx'
        data_sheet_fn = 'data annotation/raw annotations (with model predictions)/for validation - 1000 docs - ori - MIMIC-III-rad (free text removed, with predictions).xlsx'
    else:
        assert dataset == 'Tayside'
        data_sheet_fn = 'for validation - 5000 docs - ori - tayside - rad.xlsx'
    df = pd.read_excel(data_sheet_fn,engine='openpyxl')
    final_ann_available = False
    
    if dataset == 'MIMIC-III':
        # load admission-level icd-based results 
        print('read full admission-level data sheet (for icd-based results)')
        pred_model_type = 'weak' # 'weak' or 'strong'
        icd9_matching_onto_source = 'both' # 'NZ', 'bp', or 'both' as icd9 matching onto path: (i) NZ (default, ordo-*icd10-icd9* with MoH NZ); (ii) bp (ORDO-*UMLS-icd9* with bioportal ICD-9-CM); (iii) from both NZ and bp sources, 'both'.
        exact_or_narrower_only = True # whether using exact or narrower only matching from ORDO to ICD-10 (if setting to True, this will result in a set of rare disease ICD-9 codes with higher precision but lower recall. This will affect sources of 'NZ' and 'both'.)
        admission_lvl_results_file_name = 'MIMIC-III DS-Rare-Disease-ICD9-new%s-filtered%s-%s%s.xlsx' % ('-rad' if data_category == 'Radiology' else '', '-sup' if pred_model_type == 'strong' else '', icd9_matching_onto_source,'-E-N' if exact_or_narrower_only else '')
        df_admission = pd.read_excel(admission_lvl_results_file_name, engine='openpyxl')
    
    # 2. filtering the data to evaluate - uncomment some of the options below
    
    # number of mention-level data to test on
    num_sample_data = len(df)
    #num_sample_data_testing = 198
    filtering_desc = 'results on %s rad testing labelled data' % len(df)
    df_filtered = df
    print(len(df_filtered))#,df_filtered)
    
    # ontology filtering setting
    auto_onto_filtering = True # whether use the rule-based onto filtering (True) or use the gold standard onto filtering (False).
    onto_matching_pred_col_name = 'ORDOisNotGroupOfDisorder' if auto_onto_filtering else 'gold UMLS-to-ORDO label'
    # set gold column names
    t2U_col_name = 'gold text-to-UMLS label' if 'gold text-to-UMLS label' in df_filtered.columns.tolist() else 'manual label from ann1'
    print('t2U_col_name:',t2U_col_name)
    
    # 3. create multi-label prediction and gold result matrices
    # get patient-stay-level results from mention-level results
    # get dict of possible ORDO diseases (with a list sorted by frequency) - a larger set of non-occurring rare diseases can affect macro-level metrics, but not micro-level metrics
    # and dict/list of all testing documents considered (order as the same in the sheet)
    dict_ORDO_relevant = defaultdict(int)
    #dict_ORDO_relevant_gold = defaultdict(int)
    #dict_doc_row_ind = defaultdict(int)
    list_doc_row_ind = []
    for index, row in df_filtered.iterrows():
        #dict_ORDO_relevant[row['ORDO with desc']] += 1
        
        if row[t2U_col_name] == 1 and row['gold UMLS-to-ORDO label'] == 1:
            dict_ORDO_relevant[row['ORDO with desc']] += 1
        else:
            dict_ORDO_relevant[row['ORDO with desc']] = 0
        #dict_doc_row_ind[row['doc row ID']] += 1
        doc_row_id_str = str(row['doc row ID'])
        if doc_row_id_str not in list_doc_row_ind:
            list_doc_row_ind.append(doc_row_id_str)
    list_ORDO_relevant_sorted = sorted(dict_ORDO_relevant, key=dict_ORDO_relevant.get, reverse=True) # ORDO ID sorted by frequency
    #list_ORDO_relevant_freq_sorted = [dict_ORDO_relevant[ORDO_ID] for ORDO_ID in list_ORDO_relevant_sorted] # the freq of the corresponding ORDO IDs
    
    #plot the label distribution figure in the testing data
    plot_label_distribution(dict_ORDO_relevant,'%s-testing-label-dist.png' % filtering_desc, dataset=dataset + '-' + data_category)
    
    # create doc-ORDO binary matrix
    np_doc_ORDO_bin_mat_gold = np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))) # initialise gold binary matrix
    
    #NLP-based doc-ORDO bin matrix
    np_doc_ORDO_bin_mat_rule_based_weak, np_doc_ORDO_bin_mat_SemEHR, np_doc_ORDO_bin_mat_men_len, np_doc_ORDO_bin_mat_prevalence, np_doc_ORDO_bin_mat_rule_and, np_doc_ORDO_bin_mat_rule_or,np_doc_ORDO_bin_mat_rule_or_in_domain_tuned, np_doc_ORDO_bin_mat_model, np_doc_ORDO_bin_mat_ds_model,np_doc_ORDO_bin_mat_model_sup,np_doc_ORDO_bin_mat_model_trained,np_doc_ORDO_bin_mat_model_trained_tuned_recall,np_doc_ORDO_bin_mat_model_trained_tuned_f1 = np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))) # initialise prediction binary matrices
    #print(np_doc_ORDO_bin_mat_rule_based_weak)
    
    for index, row in df_filtered.iterrows():
        doc_row_id_str = str(row['doc row ID'])
        ORDO_with_desc = row['ORDO with desc']
        mat_row_id = list_doc_row_ind.index(doc_row_id_str) # get the row id of the matrix
        mat_column_id = list_ORDO_relevant_sorted.index(ORDO_with_desc) # get the column id of the matrix
        np_doc_ORDO_bin_mat_gold[mat_row_id,mat_column_id] = 1 if row[t2U_col_name] == 1 and row['gold UMLS-to-ORDO label'] == 1 else 0
        
        #update the doc-ORDO bin matrices to 1 when the two predictions (text-UMLS, UMLS-ORDO) are True/1
        if row['pos label: both rules applied'] == 1 and row[onto_matching_pred_col_name] == 1: 
            np_doc_ORDO_bin_mat_rule_based_weak[mat_row_id,mat_column_id] = 1
        if row['SemEHR label'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_SemEHR[mat_row_id,mat_column_id] = 1 
        if row['rule (mention length >3)'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_men_len[mat_row_id,mat_column_id] = 1
        if row['rule (prevalance th <= 0.005)'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_prevalence[mat_row_id,mat_column_id] = 1
        if row['rule (mention length >3)'] == 1 and row['rule (prevalance th <= 0.005) - transferred'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_rule_and[mat_row_id,mat_column_id] = 1
        if (row['rule (mention length >3)'] == 1 or row['rule (prevalance th <= 0.005) - transferred'] == 1) and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_rule_or[mat_row_id,mat_column_id] = 1
        if (row['rule (mention length >4)'] == 1 or row['rule (prevalance th <= 0.0005)'] == 1) and row[onto_matching_pred_col_name] == 1:
            #SemEHR + rule-based results (with rules tuned with in-domain data) 
            np_doc_ORDO_bin_mat_rule_or_in_domain_tuned[mat_row_id,mat_column_id] = 1    
        if row['model blueBERTnorm prediction'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_model[mat_row_id,mat_column_id] = 1
        if dataset == 'MIMIC-III':
            if row['model blueBERTnorm prediction ds'] == 1 and row[onto_matching_pred_col_name] == 1:
                np_doc_ORDO_bin_mat_ds_model[mat_row_id,mat_column_id] = 1 
        if row['model blueBERTnorm prediction sup'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_model_sup[mat_row_id,mat_column_id] = 1
        if row['model blueBERTnorm prediction p0.005 l3'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_model_trained[mat_row_id,mat_column_id] = 1
        if row['model blueBERTnorm prediction p0.01 l4'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_model_trained_tuned_recall[mat_row_id,mat_column_id] = 1
        if row['model blueBERTnorm prediction p0.0005 l4'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_model_trained_tuned_f1[mat_row_id,mat_column_id] = 1
    
    #ICD-based doc-ORDO bin matrix - for data with ICD codes only (e.g. MIMIC-III)
    np_doc_ORDO_bin_mat_ICD, np_doc_ORDO_bin_mat_NLP, np_doc_ORDO_bin_mat_ICD_or_NLP= np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant)))
    if dataset == 'MIMIC-III':
        dict_row_id_to_ICD_list = {}
        pattern = "'(.*?)'"
        for mat_row_id, doc_row_id_ in tqdm(enumerate(list_doc_row_ind)):
            #get the matched ORDO from ICD9 codes
            doc_row_id_ = doc_row_id_[1:-1] if doc_row_id_[:1] == '[' and doc_row_id_[-1:] == ']' else doc_row_id_ # if there's squared bracket, remove them.
            df_adm_matched_row_id = df_admission[df_admission['ROW_ID'].astype(str)==str(doc_row_id_)]
            matched_ORDOs_by_ICD9_list_as_str = df_adm_matched_row_id['ORDO_ID_icd9_manual'].to_string(index=False) # string of a list form
            matched_ORDOs_by_ICD9_list = re.findall(pattern,matched_ORDOs_by_ICD9_list_as_str) # get the list by extracting '' in the string of a list form
            if doc_row_id_ == '[26976]':
                print('ORDO_IDs for doc [26976]:', matched_ORDOs_by_ICD9_list_as_str)
            #get the column index and update the doc-ORDO bin matrix
            for matched_ORDO in matched_ORDOs_by_ICD9_list:
                for ind_, ORDO_with_desc in enumerate(list_ORDO_relevant_sorted):
                    if ORDO_with_desc.startswith(matched_ORDO + ' '):
                        mat_column_id = ind_
                        np_doc_ORDO_bin_mat_ICD[mat_row_id,mat_column_id] = 1
                        break
                        
            #get the matched ORDO from NLP
            matched_ORDOs_by_NLP_list_as_str = df_adm_matched_row_id['ORDO_ID_filtered'].to_string(index=False) # string of a list form
            matched_ORDOs_by_NLP = re.findall(pattern,matched_ORDOs_by_NLP_list_as_str) # get the list by extracting '' in the string of a list form
            #get the column index and update the doc-ORDO bin matrix
            for matched_ORDO in matched_ORDOs_by_NLP:
                for ind_, ORDO_with_desc in enumerate(list_ORDO_relevant_sorted):
                    if ORDO_with_desc.startswith(matched_ORDO + ' '):
                        mat_column_id = ind_
                        np_doc_ORDO_bin_mat_NLP[mat_row_id,mat_column_id] = 1
                        break
        
        #for radiology reports - using the tuned re-trained model as the NLP model
        #optimised for Recall
        #np_doc_ORDO_bin_mat_NLP = np_doc_ORDO_bin_mat_model_trained_tuned_recall
        #optimised for F1
        np_doc_ORDO_bin_mat_NLP = np_doc_ORDO_bin_mat_model_trained_tuned_f1
        np_doc_ORDO_bin_mat_ICD_or_NLP = np.clip(np_doc_ORDO_bin_mat_ICD + np_doc_ORDO_bin_mat_NLP,0,1)
            
        #some checks for the generated binary matrices
        #np.set_printoptions(threshold=sys.maxsize) # to print large matrix without ellipses
        print('np_doc_ORDO_bin_mat_gold:', np_doc_ORDO_bin_mat_gold.shape, np.sum(np_doc_ORDO_bin_mat_gold, axis=None), np_doc_ORDO_bin_mat_gold)
        #check (i): output the "false positive" of icd-based by comparing icd-based bin mat to the gold bin mat
        compare_element_bin = np_doc_ORDO_bin_mat_ICD == np_doc_ORDO_bin_mat_gold
        compare_element_bin_diff_ind = np.argwhere(compare_element_bin==False)
        list_ICD_fp_cases = []
        for i,j in compare_element_bin_diff_ind:
            if np_doc_ORDO_bin_mat_ICD[i,j] == 1 and np_doc_ORDO_bin_mat_gold[i,j] == 0:
                print('ICD \"false positive\" elements',list_doc_row_ind[i] + ' ' + list_ORDO_relevant_sorted[j])   
                list_ICD_fp_cases.append([list_doc_row_ind[i][1:-1],list_ORDO_relevant_sorted[j]])
        df_ICD_fp_cases = pd.DataFrame(list_ICD_fp_cases,columns=['doc row ID','ORDO ID by ICD but not in gold ann']) # from a 2D-list to a pandas dataframe
        df_ICD_fp_cases.to_csv('ICD-false-positive-cases-%s%s.csv' % (icd9_matching_onto_source,'-E-N' if exact_or_narrower_only else ''),index=False) # after this output is generated, the full texts can be matched to the doc row IDs with vloopuk on Excel, by linking to a sheet with full text (generated by step9_processing_all_documents.py with save_full_text option set as True, e.g. MIMIC-III DS-Rare-Disease-ICD9-new.txt). The final generated output was called ICD-false-positive-cases-both-E-N-with-full-texts.txt. We do not implement this step by code here as this is not related to the main function this program.
        #check (ii): output the diff between the bin mat aggregated from mention and the bin mat directly from the admission-level sheet for a check if they are different
        try:
            np_doc_ORDO_bin_mat_NLP_from_mention = np_doc_ORDO_bin_mat_model_sup if pred_model_type == 'strong' else np_doc_ORDO_bin_mat_model
            np_doc_ORDO_bin_mat_NLP_from_admission = np_doc_ORDO_bin_mat_NLP
            assert np.array_equal(np_doc_ORDO_bin_mat_NLP_from_mention,np_doc_ORDO_bin_mat_NLP_from_admission)
        except AssertionError:
            #df_doc_ORDO_bin_mat_NLP_from_mention = pd.DataFrame(np_doc_ORDO_bin_mat_NLP_from_mention,index=list_doc_row_ind,columns=list_ORDO_relevant_sorted)
            #df_doc_ORDO_bin_mat_NLP_from_mention.to_csv('doc_ORDO_bin_mat_NLP_from_mention.csv')
            #df_doc_ORDO_bin_mat_NLP_from_admission = pd.DataFrame(np_doc_ORDO_bin_mat_NLP_from_admission,index=list_doc_row_ind,columns=list_ORDO_relevant_sorted)
            #df_doc_ORDO_bin_mat_NLP_from_admission.to_csv('doc_ORDO_bin_mat_NLP_from_admission.csv')
            compare_element_bin = np_doc_ORDO_bin_mat_NLP_from_mention == np_doc_ORDO_bin_mat_NLP_from_admission
            compare_element_bin_diff_ind = np.argwhere(compare_element_bin==False)
            for i,j in compare_element_bin_diff_ind:
                print('diff elements',list_doc_row_ind[i] + ' ' + list_ORDO_relevant_sorted[j])
                print('np_doc_ORDO_bin_mat_NLP_from_mention:',np_doc_ORDO_bin_mat_NLP_from_mention[i,j])
                print('np_doc_ORDO_bin_mat_NLP_from_admission:',np_doc_ORDO_bin_mat_NLP_from_admission[i,j])
                
    # 4. get prediction results and calculate precision, recall, and F1
    # also export the results to a csv file
    #print(all_macro(np_doc_ORDO_bin_mat_model,np_doc_ORDO_bin_mat_gold))
    #print(all_micro(np_doc_ORDO_bin_mat_model.ravel(),np_doc_ORDO_bin_mat_gold.ravel())) # (0.46153846153846156, 0.5714285714285714, 0.7058823529411765, 0.6315789473684211)
    
    list_result_desc = ['rule-based weakly annotation results', 'SemEHR results', 'mention length rule results', 'prevalence rule results', 'both rule AND results', 'both rule OR results', 'both rule OR results in-domain tuned', 'model non-masked results', 'model non-masked ds results', 'full supervised model blueBERTnorm prediction results', 'model non-masked results trained', 'model non-masked results trained tuned by recall', 'model non-masked results trained tuned by f1']
    if dataset == 'MIMIC-III':
        icd_and_nlp_results = ['icd-based', 'nlp%s-based' % ('(sup)' if pred_model_type == 'strong' else ''), 'nlp%s+icd-based' % ('(sup)' if pred_model_type == 'strong' else '')]
        icd_and_nlp_results = [result_str + ' %s%s results' % (icd9_matching_onto_source,'-E-N' if exact_or_narrower_only else '') for result_str in icd_and_nlp_results]
        list_result_desc = list_result_desc + icd_and_nlp_results
            
    for ind, np_doc_ORDO_bin_mat_pred in enumerate([np_doc_ORDO_bin_mat_rule_based_weak, np_doc_ORDO_bin_mat_SemEHR, np_doc_ORDO_bin_mat_men_len, np_doc_ORDO_bin_mat_prevalence, np_doc_ORDO_bin_mat_rule_and, np_doc_ORDO_bin_mat_rule_or, np_doc_ORDO_bin_mat_rule_or_in_domain_tuned, np_doc_ORDO_bin_mat_model, np_doc_ORDO_bin_mat_ds_model, np_doc_ORDO_bin_mat_model_sup,np_doc_ORDO_bin_mat_model_trained,np_doc_ORDO_bin_mat_model_trained_tuned_recall,np_doc_ORDO_bin_mat_model_trained_tuned_f1,np_doc_ORDO_bin_mat_ICD,np_doc_ORDO_bin_mat_NLP,np_doc_ORDO_bin_mat_ICD_or_NLP]):
        # no ds results for Tayside - simply continue the loop
        if dataset == 'Tayside' and ind >= len(list_result_desc): 
            # break when it starts reach icd_and_nlp_results part for the Tayside data
            break
        if dataset == 'Tayside' and list_result_desc[ind] == 'model non-masked ds results':
            continue
        
        #print(np.array_equal(np_doc_ORDO_bin_mat_pred,np_doc_ORDO_bin_mat_rule_based_weak))
        display_and_save_multi_label_results(np_doc_ORDO_bin_mat_pred,np_doc_ORDO_bin_mat_gold,list_ORDO_relevant_sorted, result_desc = filtering_desc + ' ' + list_result_desc[ind] + '.xlsx', display_per_label_results=display_per_label_results, output_xlsx=output_xlsx)