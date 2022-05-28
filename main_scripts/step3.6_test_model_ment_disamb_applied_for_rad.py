#!/usr/bin/python
# coding=utf-8

# only test the phenotype confirmation model trained from MIMIC-III discharge summaries for new documents.

from sent_bert_emb_viz_util import load_data, encode_data_tuple, test_model_from_encoding_output
import pickle
import os, sys
from step4_further_results_from_annotations import get_and_display_results
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import argparse

#to shut down the server with commandline: bert-serving-terminate -port 5555

if __name__ == '__main__':
#0. setting
    parser = argparse.ArgumentParser(description="step3 - apply pre-trained phenotype confirmation models to new radiology reports")
    parser.add_argument('-d','--dataset', type=str,
                    help="dataset name", default='Tayside')
    parser.add_argument('-dc','--data-category', type=str,
                    help="category of data", default='Discharge summary')
    parser.add_argument('-st','--supervision-type', type=str,
                    help="type of supervision: weak or strong", default='weak')
    parser.add_argument('-wsm','--weak-supervision-model-path', type=str,
                    help="only used when direct transfer (i.e. --trans), weak supervision model path, default as trained from MIMIC-III discharge summaries", default='./models/model_blueBERTnorm_ws5.pik') 
    parser.add_argument('-ssm','--strong-supervision-model-path', type=str,
                    help="strong supervision model path, default as trained from manual annotation (validation data) from MIMIC-III discharge summaries", default='./models/model_blueBERTnorm_ws5_sup.pik')
    parser.add_argument('-f','--fill-data', help='fill the predictions on the evaluation sheet',action='store_true',default=False)
    parser.add_argument('-e','--do-eval', help='calculate metric results based on gold annotation, if available, otherwise evaluate by treating all data are True',action='store_true',default=False)
    parser.add_argument('-m','--masked-training', help='mention masking in encoding',action='store_true',default=False)
    parser.add_argument('-ds','--use-document-structure', help='use document structure in encoding',action='store_true',default=False)
    
    args = parser.parse_args()
    
    
    dataset=args.dataset # 'MIMIC-III' or 'Tayside'
    data_category = args.data_category # 'Radiology' or 'Discharge summary'
    assert data_category == 'Radiology' # this program is about applying the model to radiology reports
    testing_data_sheet_name = 'for validation - 1000 docs - ori - MIMIC-III-rad.xlsx' if dataset == 'MIMIC-III' else 'for validation - 5000 docs - ori - %s - rad.xlsx' % dataset.lower()
    
    pred_model_type = args.supervision_type # 'weak' or 'strong'
    
    fill_data = args.fill_data
    calculate_baseline_results = args.do_eval
    
    masked_training = args.masked_training
    use_doc_struc = args.use_document_structure
    
    model_path = './bert-models/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12/'; model_name = 'blueBERTnorm'
    
    if dataset == 'MIMIC-III':
        marking_str_te = 'testing_198_MIMIC-III_rad'
    elif dataset == 'Tayside':
        marking_str_te = 'testing_279_Tayside_rad'
    else:
        marking_str_te = 'testing_%s_rad' % dataset
    
#1. load trained weak/strong supervision models
    if pred_model_type == 'weak':
        trained_model_name = args.weak_supervision_model_path
        if os.path.exists(trained_model_name):
            with open(trained_model_name, 'rb') as data_f:
                clf_non_masked_ds, _, clf_non_masked = pickle.load(data_f) # loading the non-masked model (not with document structure) to be applied for radiology reports
    elif pred_model_type == 'strong':
        trained_model_name = args.strong_supervision_model_path # using strongly supervised model
        if os.path.exists(trained_model_name):
            with open(trained_model_name, 'rb') as data_f:
                clf_non_masked_ds,clf_non_masked = pickle.load(data_f) # loading the non-masked model (not with document structure) to be applied for radiology reports
    else:
        print('pred_model_type wrong, neither weak or strong, value:%s' % pred_model_type)
        sys.exit(0)
    clf = clf_non_masked_ds if use_doc_struc else clf_non_masked
    
#2. load testing data and predict results: 
#load data from .xlsx and save the results to a specific column
    # get a list of data tuples from an annotated .xlsx file
    # data format: a 6-element tuple of section_text, document_structure_name, mention_name, UMLS_code, UMLS_desc, label (True or False)
    df = pd.read_excel(testing_data_sheet_name,engine='openpyxl')
    
    data_list_tuples = []
    for i, row in df.iterrows():
        #filter out the manually added rows that were created during the annotation
        if 'manually added data' in row:
            if not pd.isna(row['manually added data']):
                print('row %s is a manually added datum based on annotation： ignored in data_list_tuples for encoding' % i)
                continue
        doc_struc = row['document structure']
        text = row['Text']
        mention = row['mention']
        UMLS_code = row['UMLS with desc'].split()[0]
        UMLS_desc = ' '.join(row['UMLS with desc'].split()[1:])
        #label = row['manual label from ann1']
        if 'gold text-to-UMLS label' in row:
            label = row['gold text-to-UMLS label']
            label = 0 if label == -1 else label # assume that the inapplicable (-1) entries are all False.
        else:
            label = 1
        #print(label)
        data_tuple = (text,doc_struc,mention,UMLS_code,UMLS_desc,label)
        #if i<2:
        #    print(data_tuple)
        data_list_tuples.append(data_tuple)
    
    # get testing data rep and predict with the model
    output_tuple_test = encode_data_tuple(data_list_tuples, masking=masked_training, with_doc_struc=use_doc_struc, model_path=model_path, marking_str=marking_str_te, window_size=5, masking_rate=1,port_number_str='5555')
    
    _,y_pred_test,list_ind_err_test = test_model_from_encoding_output(output_tuple_test,len(data_list_tuples),clf) 
    y_pred_test = np.asarray(y_pred_test)
    
    if calculate_baseline_results:
        # filter df by rule-based annotation, if chosen to; otherwise, keep the filtered df same as df
        df_filtered = df[(df['neg label: only when both rule 0']=='') & (df['pos label: both rules applied']=='')] if test_with_non_rule_annotated_only else df
        print('df_filtered:',len(df_filtered))
        
        #print metrics
        print('SemEHR results')
        #y_pred_SemEHR_test = np.ones_like(y_test)
        y_pred_SemEHR_test = df_filtered[['SemEHR label']].to_numpy() 
        y_pred_SemEHR_test_labelled = [y_pred_SemEHR_test[ind] for ind, y_test_ele in enumerate(y_test) if y_test_ele != -1] # filtered by annotated data (same for the other results below)
        print('test precision:', precision_score(y_test_labelled, y_pred_SemEHR_test_labelled), 'test recall:', recall_score(y_test_labelled, y_pred_SemEHR_test_labelled), 'test F1:', f1_score(y_test_labelled, y_pred_SemEHR_test_labelled))
        
        print('mention length rule results')
        y_pred_ment_len = df_filtered[['rule (mention length >3)']].to_numpy()
        y_pred_ment_len_labelled = [y_pred_ment_len[ind] for ind, y_test_ele in enumerate(y_test) if y_test_ele != -1]
        print('test precision:', precision_score(y_test_labelled, y_pred_ment_len_labelled), 'test recall:', recall_score(y_test_labelled, y_pred_ment_len_labelled), 'test F1:', f1_score(y_test_labelled, y_pred_ment_len_labelled))
        
        print('prevalence rule results')
        y_pred_prevalence = df_filtered[['rule (prevalance th <= 0.005)']].to_numpy()
        y_pred_prevalence_labelled = [y_pred_prevalence[ind] for ind, y_test_ele in enumerate(y_test) if y_test_ele != -1]
        print('test precision:', precision_score(y_test_labelled, y_pred_prevalence_labelled), 'test recall:', recall_score(y_test_labelled, y_pred_prevalence_labelled), 'test F1:', f1_score(y_test_labelled, y_pred_prevalence_labelled))
        
        print('both rule AND results')
        y_pred_rules_labelled = np.logical_and(y_pred_ment_len_labelled,y_pred_prevalence_labelled).astype(int)
        #y_pred_rules = [y_pred_rules[ind] for ind, y_test_ele in enumerate(y_test) if y_test_ele != -1]
        print('test precision:', precision_score(y_test_labelled, y_pred_rules_labelled), 'test recall:', recall_score(y_test_labelled, y_pred_rules_labelled), 'test F1:', f1_score(y_test_labelled, y_pred_rules_labelled))
        
        print('both rule OR results')
        y_pred_rules_labelled = np.logical_or(y_pred_ment_len_labelled,y_pred_prevalence_labelled).astype(int)
        #y_pred_rules = [y_pred_rules[ind] for ind, y_test_ele in enumerate(y_test) if y_test_ele != -1]
        print('test precision:', precision_score(y_test_labelled, y_pred_rules_labelled), 'test recall:', recall_score(y_test_labelled, y_pred_rules_labelled), 'test F1:', f1_score(y_test_labelled, y_pred_rules_labelled))
       
    if fill_data:
        print('df_length:',len(df))
        print('y_pred_test:',y_pred_test.shape)
        #fill the prediction into the .xlsx file
        ind_y_pred_test=0
        result_column_name = 'model %s prediction%s%s%s' % (model_name, ' (masked training)' if masked_training else '', ' ds' if use_doc_struc else '', ' sup' if pred_model_type == 'strong' else '')
        print('updating data (if necessary) for %s' % result_column_name)
        if not result_column_name in df.columns:
            df[result_column_name] = ""
        ind_non_manual = 0
        for i, row in df.iterrows():
            #filter out the manually added rows that were created during the annotation (for Tayside data)
            if 'manually added data' in row:
                if not pd.isna(row['manually added data']):
                    df.at[i,result_column_name] = 0 # set the pred as 0 if this is a manually added row  
                    print('row %s is a manually added datum based on annotation' % i)
                    continue
            if ind_non_manual in list_ind_err_test:
                df.at[i,result_column_name] = 0 # set the pred as 0 if there is an error (we use this rule just for Tayside data)                
                print('row %s results set to 0 due to encoding error' % str(i))
                ind_non_manual = ind_non_manual + 1
                continue
            if row[result_column_name] != y_pred_test[ind_y_pred_test]:
                print('row %s results changed %s to %s' % (str(i), row[result_column_name], y_pred_test[ind_y_pred_test]))
            df.at[i,result_column_name] = y_pred_test[ind_y_pred_test]
            ind_non_manual = ind_non_manual + 1
            ind_y_pred_test = ind_y_pred_test + 1
        df.to_excel(testing_data_sheet_name[:len(testing_data_sheet_name)-len('.xlsx')] + ' - predicted%s%s%s.xlsx' % (' - masked' if masked_training else '', ' - ds' if use_doc_struc else '', ' - sup' if pred_model_type == 'strong' else ''),index=False)