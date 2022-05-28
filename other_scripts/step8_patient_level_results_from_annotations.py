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

# the evaluation functions of micro/macro/instance accuracy/precision/recall/F1 are adapted from CAML-MIMIC https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py under the MIT License

# run with python -W ignore step8.1_patient_level_results_from_annotations.py if there are many warnings (UndefinedMetricWarning and RuntimeWarning).

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

# avoid ... in showing long sequence
pd.set_option("display.max_colwidth", 10000) # this is necessary for retrieving long str from dataframe!

def plot_label_distribution(dict_concept_freq,filename,dataset='MIMIC-III'):

    list_ORDO_relevant_gold_sorted = sorted(dict_concept_freq, key=dict_concept_freq.get, reverse=True) # ORDO ID sorted by frequency
    list_ORDO_relevant_gold_freq_sorted = [dict_concept_freq[ORDO_ID] for ORDO_ID in list_ORDO_relevant_gold_sorted] # the freq of the corresponding ORDO IDs
    
    list_x_axis = [ele for ele in range(1,len(list_ORDO_relevant_gold_freq_sorted)+1)]
    plt.title('Distribution of labels for the %s-ORDO-testing dataset' % dataset)
    plt.plot(list_x_axis,list_ORDO_relevant_gold_freq_sorted,'k+',label='MIMIC-III-ORDO-testing')
    plt.xlabel('Index of ORDO code in %s-ORDO-testing' % dataset)
    plt.ylabel('Number of occurrences')
    #plt.legend(loc='upper right')
    plt.xticks(np.arange(0, len(list_ORDO_relevant_gold_freq_sorted)+1, 5))
    # add text label to the figure
    top_k_ORDO_ID_to_display = 5 # the most freq ORDO IDs to display
    texts = [plt.text(list_x_axis[i], list_ORDO_relevant_gold_freq_sorted[i], list_ORDO_relevant_gold_sorted[i], ha='center', va='center', fontsize=6) for i in range(top_k_ORDO_ID_to_display)]
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.savefig(filename,dpi=300)
    
def get_and_display_ORDO_results(y_true, y_pred_UMLS,y_pred_ORDO):
    # get text-to-ORDO predictions from text-to-UMLS and UMLS-to-ORDO predictions
    y_pred = np.multiply(y_pred_UMLS,y_pred_ORDO)
    # calculate precision, recall, and F1, and display confusion matrix
    prec, rec, f1 = precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print('test precision: %s test recall: %s test F1: %s' % (str(prec), str(rec), str(f1)))
    print('tp %s tn %s fp %s fn %s\n' % (str(tp),str(tn),str(fp),str(fn)))
    
# *multi-label classification metrics*

# display per-label precision, recall, F1, and confusion matrix, and micro-level results.
'''
list_label: a list of label corresponding to the rows in yhat and y
output_xlsx: whether to output the result to excel sheets
'''
def display_and_save_multi_label_results(yhat, y, list_label, result_desc='', display_per_label_results=False, output_xlsx=False, float_digit=3):
    if result_desc != '': 
        print(result_desc)
    #to create a data frame: set column and index name and save the results to a numpy array 
    column_name = ['tp', 'tn', 'fp', 'fn', 'tp+fn', 'acc', 'prec', 'rec', 'F1']
    index_name = list_label + ['overall (micro-level)','overall (macro-level)','overall (instance-level)']
    np_arr_results = np.zeros((len(index_name),len(column_name)))
    for i in range(len(list_label)):
        y_true = y[:,i]
        y_pred = yhat[:,i]
        acc, prec, rec, f1 = accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred) # here can be zero_devision error as y_true or y_pred can be vector of zeros.
        #print(y_true,y_pred)
        #print(confusion_matrix(y_true, y_pred))
        if np.isin(1,y_true) or np.isin(1,y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()[0], 0, 0, 0
        
        np_arr_results[i,:] = np.array([tp,tn,fp,fn,tp+fn,acc,prec,rec,f1])
        if display_per_label_results:
            print('%s: tp %s tn %s fp %s fn %s accuracy %s precision %s recall %s F1 %s' % (list_label[i],str(tp),str(tn),str(fp),str(fn),str(acc), str(prec), str(rec), str(f1)))
    micro_acc, micro_prec, micro_rec, micro_f1 = micro_accuracy(yhat.ravel(),y.ravel()), micro_precision(yhat.ravel(),y.ravel()), macro_recall(yhat.ravel(),y.ravel()), macro_f1_score(yhat.ravel(),y.ravel())
    micro_tn, micro_fp, micro_fn, micro_tp = confusion_matrix(y.ravel(), yhat.ravel()).ravel()
    macro_acc, macro_prec, macro_rec, macro_f1 = macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1_score(yhat, y)
    instance_acc,instance_prec,instance_rec,instance_f1 = inst_accuracy(yhat, y), inst_precision(yhat, y), inst_recall(yhat, y), inst_f1(yhat, y) # here can have RuntimeWarning error as rows in yhat and/or y can be vector of zeros.
    print('micro-level accurarcy %s precision %s recall %s F1 %s' % (str(round(micro_acc,float_digit)), str(round(micro_prec,float_digit)), str(round(micro_rec,float_digit)), str(round(micro_f1,float_digit))))
    print('macro-level accurarcy %s precision %s recall %s F1 %s' % (str(round(macro_acc,float_digit)), str(round(macro_prec,float_digit)), str(round(macro_rec,float_digit)), str(round(macro_f1,float_digit))))
    print('instance-level accurarcy %s precision %s recall %s F1 %s' % (str(round(instance_acc,float_digit)), str(round(instance_prec,float_digit)), str(round(instance_rec,float_digit)), str(round(instance_f1,float_digit))))
    print('overall tp %s tn %s fp %s fn %s\n' % (str(micro_tp),str(micro_tn),str(micro_fp),str(micro_fn)))
    np_arr_results[-3,:] = np.array([micro_tp,micro_tn,micro_fp,micro_fn,None,micro_acc, micro_prec, micro_rec, micro_f1])
    np_arr_results[-2,:] = np.array([None,None,None,None,None,macro_acc, macro_prec, macro_rec, macro_f1])
    np_arr_results[-1,:] = np.array([None,None,None,None,None,instance_acc,instance_prec,instance_rec,instance_f1])
    
    if output_xlsx:
        df_results = pd.DataFrame(np_arr_results, index=index_name, columns=column_name)
        df_results.to_excel(result_desc,index=True)
        
# the code below is from https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py under the MIT license
def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)

def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic, ymic)

#########################################################################
#MACRO METRICS: calculate metric for each label and average across labels
#########################################################################

def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)

def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_f1_score(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1
    
##########################################################################
#MICRO METRICS: treat every prediction as an individual binary prediction
##########################################################################

def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)

def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

###################
# INSTANCE-AVERAGED
###################

def inst_accuracy(yhat, y):
    num = intersect_size(yhat, y, 1) / union_size(yhat, y, 1)
    #correct for divide-by-zeros
    #num[np.isnan(num)] = 0.
    #return np.mean(num)
    return np.nanmean(num) 
    # np.nanmean: Compute the arithmetic mean along the specified axis, ignoring NaNs.
    
def inst_precision(yhat, y):
    num = intersect_size(yhat, y, 1) / yhat.sum(axis=1)
    #correct for divide-by-zeros
    #num[np.isnan(num)] = 0.
    #return np.mean(num)
    return np.nanmean(num)
    
def inst_recall(yhat, y):
    num = intersect_size(yhat, y, 1) / y.sum(axis=1)
    #correct for divide-by-zeros
    #num[np.isnan(num)] = 0.
    #return np.mean(num)
    return np.nanmean(num)
    
def inst_f1(yhat, y):
    prec = inst_precision(yhat, y)
    rec = inst_recall(yhat, y)
    f1 = 2*(prec*rec)/(prec+rec)
    return f1
    
def union_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)
    #numpy.logical_and(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'logical_and'>
    #Compute the truth value of x1 AND x2 element-wise.

if __name__ == '__main__':
    # 0. display and output setting
    display_per_label_results = False
    # output the results to excel - each method to a sheet
    output_xlsx = True
    
    # 1. load binary prediction results from sheet
    data_sheet_fn = 'data annotation/raw annotations (with model predictions)/for validation - SemEHR ori (MIMIC-III-DS, free text removed, with predictions).xlsx'
    df = pd.read_excel(data_sheet_fn,engine='openpyxl')
    
    # load admission-level icd-based results 
    pred_model_type = 'weak' # 'weak' or 'strong'
    icd9_matching_onto_source = 'both' # 'NZ', 'bp', or 'both' as icd9 matching onto path: (i) NZ (default, ordo-*icd10-icd9* with MoH NZ); (ii) bp (ORDO-*UMLS-icd9* with bioportal ICD-9-CM); (iii) from both NZ and bp sources, 'both'.
    exact_or_narrower_only = True # whether using exact or narrower only matching from ORDO to ICD-10 (if setting to True, this will result in a set of rare disease ICD-9 codes with higher precision but lower recall. This will affect sources of 'NZ' and 'both'.)
    admission_lvl_results_file_name = 'MIMIC-III DS-Rare-Disease-ICD9-new-filtered%s-%s%s.xlsx' % ('-sup' if pred_model_type == 'strong' else '', icd9_matching_onto_source,'-E-N' if exact_or_narrower_only else '')
    df_admission = pd.read_excel(admission_lvl_results_file_name,engine='openpyxl')
    
    # 2. filtering the data to evaluate - uncomment some of the options below
    
    # number of mention-level data to test on
    num_sample_data = 400 #400 for validation data; len(df) for all evaluation data
    num_sample_data_testing = 673 #673 for testing data

    # get results of the first k data instances
    #df = df[:num_sample_data]; filtering_desc = 'results on %s labelled data' % len(df)
    df = df[-num_sample_data_testing:]; filtering_desc = 'results on %s testing labelled data' % len(df)
    df_filtered = df
    
    # get results of rule-labelled ones which are also manually labelled
    #df_filtered = df[((~ df['neg label: only when both rule 0'].isna()) | (~ df['pos label: both rules applied'].isna()))]# & (~ df['manual label from ann1'].isna())]; 
    #filtering_desc = 'results on %s seen of %s labelled data' % (len(df_filtered), len(df))
    
    # get results of unseen ones
    #df_filtered = df[(df['neg label: only when both rule 0'].isna()) & (df['pos label: both rules applied'].isna())]# & (~ df['manual label from ann1'].isna())]; 
    #filtering_desc = 'results on %s unseen of %s labelled data' % (len(df_filtered), len(df))
    
    print(len(df_filtered))#,df_filtered)
    
    # ontology filtering setting
    auto_onto_filtering = True # whether use the rule-based onto filtering (True) or use the gold standard onto filtering (False).
    onto_matching_pred_col_name = 'ORDOisNotGroupOfDisorder' if auto_onto_filtering else 'gold UMLS-to-ORDO label'
        
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
        if row['gold text-to-UMLS label'] == 1 and row['gold UMLS-to-ORDO label'] == 1:
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
    plot_label_distribution(dict_ORDO_relevant,'%s-testing-label-dist.png' % filtering_desc)
    
    # create doc-ORDO binary matrix
    np_doc_ORDO_bin_mat_gold = np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))) # initialise gold binary matrix
    
    #NLP-based doc-ORDO bin matrix
    np_doc_ORDO_bin_mat_rule_based_weak, np_doc_ORDO_bin_mat_g_api_cw5_new, np_doc_ORDO_bin_mat_g_api_cw5, np_doc_ORDO_bin_mat_g_api_cw10, np_doc_ORDO_bin_mat_g_api_cw50, np_doc_ORDO_bin_mat_MedCAT, np_doc_ORDO_bin_mat_SemEHR, np_doc_ORDO_bin_mat_men_len, np_doc_ORDO_bin_mat_prevalence, np_doc_ORDO_bin_mat_rule_and, np_doc_ORDO_bin_mat_rule_or, np_doc_ORDO_bin_mat_model, np_doc_ORDO_bin_mat_ds_model, np_doc_ORDO_bin_mat_ds_model_data_sel, np_doc_ORDO_bin_mat_masked_model, np_doc_ORDO_bin_mat_masked_ds_model, np_doc_ORDO_bin_mat_ensem_best_scenario,np_doc_ORDO_bin_mat_ds_model_sup = np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))) # initialise prediction binary matrices
    #print(np_doc_ORDO_bin_mat_rule_based_weak)

    for index, row in df_filtered.iterrows():
        doc_row_id_str = str(row['doc row ID'])
        ORDO_with_desc = row['ORDO with desc']
        mat_row_id = list_doc_row_ind.index(doc_row_id_str) # get the row id of the matrix
        mat_column_id = list_ORDO_relevant_sorted.index(ORDO_with_desc) # get the column id of the matrix
        np_doc_ORDO_bin_mat_gold[mat_row_id,mat_column_id] = 1 if row['gold text-to-UMLS label'] == 1 and row['gold UMLS-to-ORDO label'] == 1 else 0
        
        #update the doc-ORDO bin matrices to 1 when the two predictions (text-UMLS, UMLS-ORDO) are True/1
        if row['pos label: both rules applied'] == 1 and row[onto_matching_pred_col_name] == 1: 
            np_doc_ORDO_bin_mat_rule_based_weak[mat_row_id,mat_column_id] = 1
        if row['Google Healthcare API cw5 new'] == 1 and row[onto_matching_pred_col_name] == 1:
            #the best G-API setting after parameter tuning 
            np_doc_ORDO_bin_mat_g_api_cw5_new[mat_row_id,mat_column_id] = 1
        if row['Google Healthcare API cw5'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_g_api_cw5[mat_row_id,mat_column_id] = 1
        if row['Google Healthcare API cw10'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_g_api_cw10[mat_row_id,mat_column_id] = 1
        if row['Google Healthcare API cw50'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_g_api_cw50[mat_row_id,mat_column_id] = 1 
        if (row['MedCAT cw5 medium']>=0.2) == 1 and row[onto_matching_pred_col_name] == 1: 
            #the best MedCAT setting after parameter tuning 
            np_doc_ORDO_bin_mat_MedCAT[mat_row_id,mat_column_id] = 1 
        if row['SemEHR label'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_SemEHR[mat_row_id,mat_column_id] = 1 
        if row['rule (mention length >3)'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_men_len[mat_row_id,mat_column_id] = 1
        if row['rule (prevalance th <= 0.005)'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_prevalence[mat_row_id,mat_column_id] = 1
        if row['rule (mention length >3)'] == 1 and row['rule (prevalance th <= 0.005)'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_rule_and[mat_row_id,mat_column_id] = 1
        if (row['rule (mention length >3)'] == 1 or row['rule (prevalance th <= 0.005)'] == 1) and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_rule_or[mat_row_id,mat_column_id] = 1
        if row['model blueBERTnorm prediction'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_model[mat_row_id,mat_column_id] = 1
        if row['model blueBERTnorm prediction ds'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_ds_model[mat_row_id,mat_column_id] = 1 
        if row['model blueBERTnorm prediction ds tr9000'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_ds_model_data_sel[mat_row_id,mat_column_id] = 1
        if row['model blueBERTnorm prediction (masked training)'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_masked_model[mat_row_id,mat_column_id] = 1
        if row['model blueBERTnorm prediction (masked training) ds'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_masked_ds_model[mat_row_id,mat_column_id] = 1
        if row['model ensemble best scenario'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_ensem_best_scenario[mat_row_id,mat_column_id] = 1
        if row['full supervised model blueBERTnorm prediction ds'] == 1 and row[onto_matching_pred_col_name] == 1:
            np_doc_ORDO_bin_mat_ds_model_sup[mat_row_id,mat_column_id] = 1
    
    #ICD-based doc-ORDO bin matrix
    np_doc_ORDO_bin_mat_ICD, np_doc_ORDO_bin_mat_NLP, np_doc_ORDO_bin_mat_ICD_or_NLP= np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant))),np.zeros((len(list_doc_row_ind),len(dict_ORDO_relevant)))
    
    dict_row_id_to_ICD_list = {}
    pattern = "'(.*?)'"
    for mat_row_id, doc_row_id_ in enumerate(list_doc_row_ind):
        #get the matched ORDO from ICD9 codes
        doc_row_id_ = doc_row_id_[1:-1] if doc_row_id_[:1] == '[' and doc_row_id_[-1:] == ']' else doc_row_id_ # if there's squared bracket, remove them.
        df_adm_matched_row_id = df_admission[df_admission['ROW_ID'].astype(str)==doc_row_id_]
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
                    
        #get the matched ORDO from ICD9 codes
        matched_ORDOs_by_NLP_list_as_str = df_adm_matched_row_id['ORDO_ID_filtered'].to_string(index=False) # string of a list form
        matched_ORDOs_by_NLP = re.findall(pattern,matched_ORDOs_by_NLP_list_as_str) # get the list by extracting '' in the string of a list form
        #get the column index and update the doc-ORDO bin matrix
        for matched_ORDO in matched_ORDOs_by_NLP:
            for ind_, ORDO_with_desc in enumerate(list_ORDO_relevant_sorted):
                if ORDO_with_desc.startswith(matched_ORDO + ' '):
                    mat_column_id = ind_
                    np_doc_ORDO_bin_mat_NLP[mat_row_id,mat_column_id] = 1
                    break
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
    #check (ii): output the diff between the bin mat aggregated from mention and the bin mar directly from the admission-level sheet for a check if they are different
    try:
        np_doc_ORDO_bin_mat_NLP_from_mention = np_doc_ORDO_bin_mat_ds_model_sup if pred_model_type == 'strong' else np_doc_ORDO_bin_mat_ds_model_data_sel
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
    
    list_result_desc = ['rule-based weakly annotation results', 'Google Healthcare API results cw5 new', 'Google Healthcare API results cw5', 'Google Healthcare API results cw10', 'Google Healthcare API results cw50', 'MedCAT results cw5 medium th0.2', 'SemEHR results', 'mention length rule results', 'prevalence rule results', 'both rule AND results', 'both rule OR results', 'model non-masked results', 'model non-masked ds results', 'model non-masked ds data sel results', 'model masked results', 'model masked ds results', 'model ensemble best scenario results', 'full supervised model blueBERTnorm prediction ds results']
    icd_and_nlp_results = ['icd-based', 'nlp%s-based' % ('(sup)' if pred_model_type == 'strong' else ''), 'nlp%s+icd-based' % ('(sup)' if pred_model_type == 'strong' else '')]
    icd_and_nlp_results = [result_str + ' %s%s results' % (icd9_matching_onto_source,'-E-N' if exact_or_narrower_only else '') for result_str in icd_and_nlp_results]
    list_result_desc = list_result_desc + icd_and_nlp_results
            
    for ind, np_doc_ORDO_bin_mat_pred in enumerate([np_doc_ORDO_bin_mat_rule_based_weak, np_doc_ORDO_bin_mat_g_api_cw5_new, np_doc_ORDO_bin_mat_g_api_cw5, np_doc_ORDO_bin_mat_g_api_cw10, np_doc_ORDO_bin_mat_g_api_cw50, np_doc_ORDO_bin_mat_MedCAT, np_doc_ORDO_bin_mat_SemEHR, np_doc_ORDO_bin_mat_men_len, np_doc_ORDO_bin_mat_prevalence, np_doc_ORDO_bin_mat_rule_and, np_doc_ORDO_bin_mat_rule_or, np_doc_ORDO_bin_mat_model, np_doc_ORDO_bin_mat_ds_model, np_doc_ORDO_bin_mat_ds_model_data_sel, np_doc_ORDO_bin_mat_masked_model, np_doc_ORDO_bin_mat_masked_ds_model,np_doc_ORDO_bin_mat_ensem_best_scenario,np_doc_ORDO_bin_mat_ds_model_sup,np_doc_ORDO_bin_mat_ICD,np_doc_ORDO_bin_mat_NLP,np_doc_ORDO_bin_mat_ICD_or_NLP]):
        #print(np.array_equal(np_doc_ORDO_bin_mat_pred,np_doc_ORDO_bin_mat_rule_based_weak))
        display_and_save_multi_label_results(np_doc_ORDO_bin_mat_pred,np_doc_ORDO_bin_mat_gold,list_ORDO_relevant_sorted, result_desc = filtering_desc + ' ' + list_result_desc[ind] + '.xlsx', display_per_label_results=display_per_label_results, output_xlsx=output_xlsx)