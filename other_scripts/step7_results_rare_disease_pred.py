# this program calculates 
# (i) the mention-level rare disease phenotype extraction (precision and pseudo-recall)
# (ii) the final patient-level ORDO concept extraction results by aggregating the mention-level predictions

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from step4_further_results_from_annotations import rule_based_model_ensemble
import numpy as np
import pandas as pd
import sys

def get_and_display_ORDO_results(y_true, y_pred_UMLS,y_pred_ORDO):
    #print(y_true.shape, y_pred_UMLS.shape, y_pred_ORDO.shape)
    # get text-to-ORDO predictions from text-to-UMLS and UMLS-to-ORDO predictions
    y_pred = np.multiply(y_pred_UMLS,y_pred_ORDO)
    # calculate precision, recall, and F1, and display confusion matrix
    acc, prec, rec, f1 = accuracy_score(y_true,y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print('test accuracy: %s test precision: %s test recall: %s test F1: %s' % (str(acc), str(prec), str(rec), str(f1)))
    print('tp %s tn %s fp %s fn %s\n' % (str(tp),str(tn),str(fp),str(fn)))
    
if __name__ == '__main__':
    onto_match_filtering = True # should be true in the real deployment
    use_gold_onto_matching = False # should be false in the real deployment
    fill_predictions = False
    
    #with_Google_API_results = False # whether display google API results as well.
    
    # 1. load binary prediction results from sheet
    data_sheet_fn = 'data annotation/raw annotations (with model predictions)/for validation - SemEHR ori (MIMIC-III-DS, free text removed, with predictions).xlsx'
    df = pd.read_excel(data_sheet_fn,engine='openpyxl')
    
    # 2. filtering the data to evaluate - uncomment some of the options below
    
    # get results of rule-labelled ones which are also manually labelled
    df = df[:400] # only evaluate the first k data in this category
    #df = df[-673:]
    
    # positive only, rule-labelled 
    #df_filtered = df[~ df['pos label: both rules applied'].isna()]# & (~ df['manual label from ann1'].isna())]
    
    # negative only, rule-labelled
    #df_filtered = df[~ df['neg label: only when both rule 0'].isna()]# & (~ df['manual label from ann1'].isna())]
    
    # all positive + negative, rule-labelled 
    #df_filtered = df[((~ df['neg label: only when both rule 0'].isna()) | (~ df['pos label: both rules applied'].isna()))]# & (~ df['manual label from ann1'].isna())]
    
    # get results of unseen ones
    # only mention length rule applied, unseen
    #df_filtered = df[(df['rule (mention length >3)']==1) & (df['rule (prevalance th <= 0.005)']==0)]# & (~ df['manual label from ann1'].isna())]
    
    # only prevalence rule applied, unseen
    #df_filtered = df[(df['rule (mention length >3)']==0) & (df['rule (prevalance th <= 0.005)']==1)]# & (~ df['manual label from ann1'].isna())]
    
    # any of the two rules applied, unseen
    #df_filtered = df[(df['neg label: only when both rule 0'].isna()) & (df['pos label: both rules applied'].isna())]# & (~ df['manual label from ann1'].isna())]
    
    # get results of the first/last k data instances
    #df_filtered = df[:250]
    #df_filtered = df[:200]
    #df_filtered = df[-673:]
    
    # no filtering
    df_filtered = df
    
    print(len(df_filtered))#,df_filtered)
    
    # 3. (1) text-to-UMLS: get gold results from the sheet
    y_test_labelled_UMLS = df_filtered[['gold text-to-UMLS label']].to_numpy()
    y_test_labelled_UMLS = np.where((y_test_labelled_UMLS==-1) | (y_test_labelled_UMLS == np.nan), 0, y_test_labelled_UMLS)
        
    # 3. (2) UMLS-to-ORDO: get gold results from the sheet
    y_test_labelled_ORDO = df_filtered[['gold UMLS-to-ORDO label']].to_numpy() 
    
    # 3. (3) text-to-ORDO as elementwise multiplication of text-to-UMLS and UMLS-to-ORDO, i.e. True only if both True.
    if onto_match_filtering:
        y_test_labelled = np.multiply(y_test_labelled_UMLS,y_test_labelled_ORDO)
    else:
        y_test_labelled = y_test_labelled_UMLS
    print(y_test_labelled.shape)
    
    # 4. get prediction results and calculate precision, recall, and F1
    # also export the results to a csv file
    
    if use_gold_onto_matching:
        y_onto_matching = y_test_labelled_ORDO
    else:
        # get the rule-based ontology matching filtering results
        y_onto_matching = df_filtered[['ORDOisNotGroupOfDisorder']].to_numpy()
    
    #print metrics
    print('rule-based weakly annotation results')
    y_rule_based_weak_labelled = np.nan_to_num(df_filtered[['pos label: both rules applied']].to_numpy())
    #simply change nans in the pos label column to 0, as neg column are all 0 or nans.
    get_and_display_ORDO_results(y_test_labelled, y_rule_based_weak_labelled,y_onto_matching)
    
    window_size = 5
    tolerance = 0
    metaAnn_mark = ''
    acc_threshold = 0.2
    model_size = 'medium'
    medcat_result_col_name = 'MedCAT cw%d %s%s%s' % (window_size,model_size,' t' + str(tolerance) if tolerance != 0 else '',metaAnn_mark)
    print('MedCAT results cw%d %s%s%s acc_th%.1f' % (window_size,model_size,' t' + str(tolerance) if tolerance != 0 else '',metaAnn_mark,acc_threshold))
    y_pred_MedCAT_test_labelled = df_filtered[[medcat_result_col_name]].to_numpy() # get the 
    y_pred_MedCAT_test_labelled_bin = y_pred_MedCAT_test_labelled >= acc_threshold
    get_and_display_ORDO_results(y_test_labelled, y_pred_MedCAT_test_labelled_bin,y_onto_matching)
                            
    print('Google Healthcare API results cw5 new') # google API results queried on 18 March 2021
    y_pred_Google_API_test_labelled = df_filtered[['Google Healthcare API cw5 new']].to_numpy() 
    get_and_display_ORDO_results(y_test_labelled, y_pred_Google_API_test_labelled,y_onto_matching)
    
    # google API results queried in Nov 2020, only 200 data were queried by the time
    # if with_Google_API_results:
        # print('Google Healthcare API results cw5')
        # y_pred_Google_API_test_labelled = df_filtered[['Google Healthcare API cw5']].to_numpy() 
        # get_and_display_ORDO_results(y_test_labelled, y_pred_Google_API_test_labelled,y_onto_matching)
        
        # print('Google Healthcare API results cw10')
        # y_pred_Google_API_test_labelled = df_filtered[['Google Healthcare API cw10']].to_numpy() 
        # get_and_display_ORDO_results(y_test_labelled, y_pred_Google_API_test_labelled,y_onto_matching)
        
        # print('Google Healthcare API results cw50')
        # y_pred_Google_API_test_labelled = df_filtered[['Google Healthcare API cw50']].to_numpy() 
        # get_and_display_ORDO_results(y_test_labelled, y_pred_Google_API_test_labelled,y_onto_matching)
    
    print('SemEHR results')
    y_pred_SemEHR_test_labelled = df_filtered[['SemEHR label']].to_numpy() 
    get_and_display_ORDO_results(y_test_labelled, y_pred_SemEHR_test_labelled,y_onto_matching)
    
    print('mention length rule results')
    y_pred_ment_len_labelled = df_filtered[['rule (mention length >3)']].to_numpy()
    get_and_display_ORDO_results(y_test_labelled, y_pred_ment_len_labelled,y_onto_matching)
    
    print('prevalence rule results')
    y_pred_prevalence_labelled = df_filtered[['rule (prevalance th <= 0.005)']].to_numpy()
    get_and_display_ORDO_results(y_test_labelled, y_pred_prevalence_labelled,y_onto_matching)
    
    print('both rule AND results')
    y_pred_rules_labelled_and = np.logical_and(y_pred_ment_len_labelled,y_pred_prevalence_labelled).astype(int)
    get_and_display_ORDO_results(y_test_labelled, y_pred_rules_labelled_and,y_onto_matching)
    
    print('both rule OR results')
    y_pred_rules_labelled_or = np.logical_or(y_pred_ment_len_labelled,y_pred_prevalence_labelled).astype(int)
    get_and_display_ORDO_results(y_test_labelled, y_pred_rules_labelled_or,y_onto_matching)

    print('model non-masked results:')
    y_pred_test_nm_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_test_nm_labelled,y_onto_matching)
    
    print('model non-masked ds results:')
    y_pred_test_nm_ds_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction ds']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_test_nm_ds_labelled,y_onto_matching)
    
    print('model non-masked ds data sel results:')
    y_pred_test_nm_ds_labelled_data_sel = np.nan_to_num(df_filtered[['model blueBERTnorm prediction ds tr9000']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_test_nm_ds_labelled_data_sel,y_onto_matching)
    
    print('model blueBERTlarge non-masked ds results:')
    y_pred_test_nm_ds_large_labelled = np.nan_to_num(df_filtered[['model blueBERTlarge prediction ds']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_test_nm_ds_large_labelled,y_onto_matching)
    
    print('model masked results:')
    y_pred_test_m_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction (masked training)']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_test_m_labelled,y_onto_matching)
    
    print('model masked ds results:')
    y_pred_test_m_ds_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction (masked training) ds']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_test_m_ds_labelled,y_onto_matching)
    
    print('model blueBERTlarge masked ds results:')
    y_pred_test_m_ds_large_labelled = np.nan_to_num(df_filtered[['model blueBERTlarge prediction (masked training) ds']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_test_m_ds_large_labelled,y_onto_matching)
    
    # rule-based model ensembling results
    print('rule-based model ensemble best scenario results:')
    #y_pred_test_m_labelled_ensemb = np.logical_or(y_pred_test_m_labelled,y_pred_test_m_ds_large_labelled).astype(int)
    y_pred_rule_based_model_ensemble_best = rule_based_model_ensemble(y_pred_ment_len_labelled, y_pred_prevalence_labelled, y_pred_test_nm_ds_labelled, y_pred_test_m_labelled)
    get_and_display_ORDO_results(y_test_labelled, y_pred_rule_based_model_ensemble_best,y_onto_matching)
    
    print('rule-based model ensemble blueBERTnorm results:')
    y_pred_rule_based_model_ensemble = rule_based_model_ensemble(y_pred_ment_len_labelled, y_pred_prevalence_labelled, y_pred_test_nm_labelled, y_pred_test_m_labelled)
    get_and_display_ORDO_results(y_test_labelled, y_pred_rule_based_model_ensemble,y_onto_matching)
    
    # print('rule-based model ensemble blueBERTlarge results:')
    # y_pred_rule_based_model_ensemble = rule_based_model_ensemble(y_pred_ment_len_labelled, y_pred_prevalence_labelled, y_pred_test_nm_labelled, y_pred_test_m_labelled)
    # get_and_display_results(y_test_labelled, y_pred_rule_based_model_ensemble)
    
    print('rule-based model ensemble ds blueBERTnorm results:')
    y_pred_rule_based_model_ensemble = rule_based_model_ensemble(y_pred_ment_len_labelled, y_pred_prevalence_labelled, y_pred_test_nm_ds_labelled, y_pred_test_m_ds_labelled)
    get_and_display_ORDO_results(y_test_labelled, y_pred_rule_based_model_ensemble,y_onto_matching)
    
    print('rule-based model ensemble ds blueBERTlarge results:')
    y_pred_rule_based_model_ensemble = rule_based_model_ensemble(y_pred_ment_len_labelled, y_pred_prevalence_labelled, y_pred_test_nm_ds_labelled, y_pred_test_m_ds_large_labelled)
    get_and_display_ORDO_results(y_test_labelled, y_pred_rule_based_model_ensemble,y_onto_matching)
    
    print('model non-masked ds results, fully supervised with 400 validation data:')
    y_pred_test_nm_ds_labelled_full_sup = np.nan_to_num(df_filtered[['full supervised model blueBERTnorm prediction ds']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_test_nm_ds_labelled_full_sup,y_onto_matching)
    
    # to export the filtered df file
    #df_filtered.to_excel('for validation - SemEHR ori - rule-labelled 135 manual results.xlsx',index=False)
    
    if fill_predictions:
        # add the predictions as a column in the sheet
        # list the method predictions and the corresponding column names separately
        list_methods_prediction = [y_test_labelled,y_pred_rules_labelled_or,y_pred_test_nm_ds_labelled,y_pred_test_nm_ds_labelled_data_sel,y_pred_rule_based_model_ensemble_best,y_pred_test_nm_ds_labelled_full_sup]
        list_column_names = ['gold test-to-ORDO label', 'rules OR ORDO', 'model blueBERTnorm prediction ds ORDO', 'model blueBERTnorm prediction ds tr9000 ORDO', 'model ensemble best scenario','model fully supervised']
        
        for method_prediction, column_name in zip(list_methods_prediction, list_column_names):
            #print(method_prediction.shape)
            #print(len(df_filtered))
            # add column headline if not there
            if not column_name in df_filtered.columns:
                df_filtered[column_name] = ""
            # update the UMLS predictions to ORDO predictions (not updating the 'gold test-to-ORDO label', which is already for ORDO)
            if column_name != 'gold test-to-ORDO label':
                method_prediction = np.multiply(method_prediction,y_onto_matching)
                #print(method_prediction.shape)
            # squezze the axes of length as 1, e.g. from (1073,1) to (1073,)    
            method_prediction = np.squeeze(method_prediction, axis=1)
            # fill the predictions
            method_pred_ind = 0
            for i, row in df_filtered.iterrows():    
                #print(method_prediction[i])
                #print(i)
                df_filtered.at[i,column_name] = method_prediction[method_pred_ind]
                method_pred_ind = method_pred_ind + 1
        df_filtered.to_excel('for validation - SemEHR ori - ORDO results added.xlsx',index=False)