# this program calculates (for radiology reports)
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
    #print('y_pred:',y_pred)
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
    dataset = 'MIMIC-III'
    if dataset == 'MIMIC-III':
        #data_sheet_fn = 'for validation - 1000 docs - ori - MIMIC-III-rad.xlsx'
        data_sheet_fn = 'data annotation/raw annotations (with model predictions)/for validation - 1000 docs - ori - MIMIC-III-rad (free text removed, with predictions).xlsx'
    else:
        assert dataset == 'Tayside'
        data_sheet_fn = 'for validation - 5000 docs - ori - tayside - rad.xlsx'
    df = pd.read_excel(data_sheet_fn,engine='openpyxl')
    final_ann_available = True
    
    df_filtered = df
    
    print(len(df_filtered))#,df_filtered)
    
    # 2. (1) text-to-UMLS: get gold results from the sheet
    if final_ann_available:
        y_test_labelled_UMLS = df_filtered[['gold text-to-UMLS label']].to_numpy()
    else:
        y_test_labelled_UMLS = df_filtered[['manual label from ann1']].to_numpy()
    y_test_labelled_UMLS = np.where((y_test_labelled_UMLS==-1) | (y_test_labelled_UMLS == np.nan), 0, y_test_labelled_UMLS)
    
    #print(y_test_labelled_UMLS.shape)
    
    # 3. (2) UMLS-to-ORDO: get gold results from the sheet
    y_test_labelled_ORDO = df_filtered[['gold UMLS-to-ORDO label']].to_numpy() 
    
    # 3. (3) text-to-ORDO as elementwise multiplication of text-to-UMLS and UMLS-to-ORDO, i.e. True only if both True.
    if onto_match_filtering:
        y_test_labelled = np.multiply(y_test_labelled_UMLS,y_test_labelled_ORDO)
    else:
        y_test_labelled = y_test_labelled_UMLS
    print(y_test_labelled.shape)
    #print('y_test_labelled:',y_test_labelled)
    
    # 4. get prediction results and calculate precision, recall, and F1
    # also export the results to a csv file
    
    if use_gold_onto_matching:
        y_onto_matching = y_test_labelled_ORDO
    else:
        # get the rule-based ontology matching filtering results
        y_onto_matching = np.nan_to_num(df_filtered[['ORDOisNotGroupOfDisorder']].to_numpy())
    
    #print metrics
    print('rule-based weakly annotation results')
    y_rule_based_weak_labelled = np.nan_to_num(df_filtered[['pos label: both rules applied - transferred']].to_numpy())
    #print('y_rule_based_weak_labelled:',y_rule_based_weak_labelled)
    #simply change nans in the pos label column to 0, as neg column are all 0 or nans.
    get_and_display_ORDO_results(y_test_labelled, y_rule_based_weak_labelled,y_onto_matching)
    
    print('SemEHR results')
    y_pred_SemEHR_test_labelled = np.nan_to_num(df_filtered[['SemEHR label']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_SemEHR_test_labelled,y_onto_matching)
    
    print('mention length rule results l3')
    y_pred_ment_len_labelled = np.nan_to_num(df_filtered[['rule (mention length >3)']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_ment_len_labelled,y_onto_matching)
    
    print('mention length rule results l4')
    y_pred_ment_len_labelled_l4 = np.nan_to_num(df_filtered[['rule (mention length >4)']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_ment_len_labelled_l4,y_onto_matching)
    
    print('prevalence rule results - transferred')
    y_pred_prevalence_labelled = np.nan_to_num(df_filtered[['rule (prevalance th <= 0.005) - transferred']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_prevalence_labelled,y_onto_matching)
    
    print('both rule AND results - transferred')
    y_pred_rules_labelled = np.logical_and(y_pred_ment_len_labelled,y_pred_prevalence_labelled).astype(int)
    get_and_display_ORDO_results(y_test_labelled, y_pred_rules_labelled,y_onto_matching)
        
    print('both rule OR results - transferred')
    y_pred_rules_labelled = np.logical_or(y_pred_ment_len_labelled,y_pred_prevalence_labelled).astype(int)
    get_and_display_ORDO_results(y_test_labelled, y_pred_rules_labelled,y_onto_matching)
    
    #for re-training
    for prevalance_th in ['0.005','0.0005','0.01']:
        print('prevalence rule results p %s' % prevalance_th)
        y_pred_prevalence_labelled = np.nan_to_num(df_filtered[['rule (prevalance th <= %s)' % prevalance_th]].to_numpy())
        get_and_display_ORDO_results(y_test_labelled, y_pred_prevalence_labelled,y_onto_matching)
        
        y_pred_ment_len_labelled = y_pred_ment_len_labelled if prevalance_th == '0.005' else y_pred_ment_len_labelled_l4
        
        print('both rule AND results p %s' % prevalance_th)
        y_pred_rules_labelled = np.logical_and(y_pred_ment_len_labelled,y_pred_prevalence_labelled).astype(int)
        get_and_display_ORDO_results(y_test_labelled, y_pred_rules_labelled,y_onto_matching)
        
        print('both rule OR results p %s' % prevalance_th)
        y_pred_rules_labelled = np.logical_or(y_pred_ment_len_labelled,y_pred_prevalence_labelled).astype(int)
        get_and_display_ORDO_results(y_test_labelled, y_pred_rules_labelled,y_onto_matching)

    print('model non-masked results:')
    y_pred_test_nm_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_test_nm_labelled,y_onto_matching)
    
    if dataset == 'MIMIC-III':
        print('model non-masked ds results:')
        y_pred_test_nm_ds_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction ds']].to_numpy())
        get_and_display_ORDO_results(y_test_labelled, y_pred_test_nm_ds_labelled,y_onto_matching)
    
    print('model non-masked fully supervised results:')
    y_pred_test_nm_sup_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction sup']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_test_nm_sup_labelled,y_onto_matching)
    
    if dataset == 'MIMIC-III':
        print('model non-masked ds fully supervised results:')
        y_pred_test_nm_ds_labelled_full_sup = np.nan_to_num(df_filtered[['model blueBERTnorm prediction ds sup']].to_numpy())
        get_and_display_ORDO_results(y_test_labelled, y_pred_test_nm_ds_labelled_full_sup,y_onto_matching)
    
    print('model non-masked results (p0.005 l3):')
    y_pred_test_nm_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction p0.005 l3']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_test_nm_labelled,y_onto_matching)
    
    print('model non-masked results (p0.01 l4):')
    y_pred_test_nm_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction p0.01 l4']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_test_nm_labelled,y_onto_matching)
        
    print('model non-masked results (p0.0005 l4):')
    y_pred_test_nm_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction p0.0005 l4']].to_numpy())
    get_and_display_ORDO_results(y_test_labelled, y_pred_test_nm_labelled,y_onto_matching)
    
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