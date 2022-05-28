# this program calculates the binary filtering of UMLS concept extraction for radiology reports from the 'for validation - 1000 docs - ori - MIMIC-III-rad.xlsx' sheet.

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import sys

# input gold labels and pred labels
# display prec,rec,F1 and confusion matrix
# also return the prec,rec,F1
def get_and_display_results(y_true, y_pred):
    prec, rec, f1 = precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print('test precision: %s test recall: %s test F1: %s' % (str(prec), str(rec), str(f1)))
    print('tp %s tn %s fp %s fn %s\n' % (str(tp),str(tn),str(fp),str(fn)))
    return prec, rec, f1

def rule_based_model_ensemble(y_pred_ment_len_labelled, y_pred_prevalence_labelled, y_pred_test_nm_labelled, y_pred_test_m_labelled):
    y_pred_rule_based_model_ensemble = np.zeros_like(y_pred_ment_len_labelled)
    #print(y_pred_ment_len_labelled.shape[0])
    for ind in range(y_pred_ment_len_labelled.shape[0]):
        if y_pred_ment_len_labelled[ind] == 1:
            if y_pred_prevalence_labelled[ind] == 1:
                # both rule applied - use non-masked training
                y_pred_rule_based_model_ensemble[ind] = y_pred_test_nm_labelled[ind]
            elif y_pred_prevalence_labelled[ind] == 0:
                # mention length > 3 only
                #y_pred_rule_based_model_ensemble[ind] = 1 if y_pred_test_nm_labelled[ind] == 1 or y_pred_test_m_labelled[ind] == 1 else 0
                y_pred_rule_based_model_ensemble[ind] = y_pred_test_nm_labelled[ind]
        elif y_pred_ment_len_labelled[ind] == 0:
            if y_pred_prevalence_labelled[ind] == 1:
                # prevalence < 0.005 only
                #y_pred_rule_based_model_ensemble[ind] = 1 if y_pred_test_nm_labelled[ind] == 1 or y_pred_test_m_labelled[ind] == 1 else 0
                y_pred_rule_based_model_ensemble[ind] = y_pred_test_nm_labelled[ind]
            elif y_pred_prevalence_labelled[ind] == 0:
                # both rule not applied
                y_pred_rule_based_model_ensemble[ind] = y_pred_test_m_labelled[ind]
                #y_pred_rule_based_model_ensemble[ind] = 1 if y_pred_test_nm_labelled[ind] == 1 or y_pred_test_m_labelled[ind] == 1 else 0
    return y_pred_rule_based_model_ensemble
    
if __name__ == '__main__':
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
    
    # 2. get gold results from the sheet
    
    if final_ann_available:
        y_test_labelled = df_filtered[['gold text-to-UMLS label']].to_numpy()
    else:
        y_test_labelled = df_filtered[['manual label from ann1']].to_numpy()
    y_test_labelled = np.where((y_test_labelled==-1) | (y_test_labelled == np.nan), 0, y_test_labelled) # 0 when not applicable or not filled
     
    print(y_test_labelled.shape)
    #print(y_test_labelled)
    #sys.exit(0)
    
    # 3. get prediction results and calculate precision, recall, and F1
    # also export the results to a csv file
    
    print('rule-based weakly annotation results')
    y_rule_based_weak_labelled = np.nan_to_num(df_filtered[['pos label: both rules applied']].to_numpy())
    #simply change nans in the pos label column to 0, as neg column are all 0 or nans.
    get_and_display_results(y_test_labelled, y_rule_based_weak_labelled)
    
    print('SemEHR results')
    y_pred_SemEHR_test_labelled = np.nan_to_num(df_filtered[['SemEHR label']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_SemEHR_test_labelled)
    
    print('mention length rule results l3')
    y_pred_ment_len_labelled = np.nan_to_num(df_filtered[['rule (mention length >3)']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_ment_len_labelled)
    
    print('mention length rule results l4')
    y_pred_ment_len_labelled_l4 = np.nan_to_num(df_filtered[['rule (mention length >4)']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_ment_len_labelled_l4)
    
    print('prevalence rule results - transferred')
    y_pred_prevalence_labelled = np.nan_to_num(df_filtered[['rule (prevalance th <= 0.005) - transferred']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_prevalence_labelled)
    
    print('both rule AND results - transferred')
    y_pred_rules_labelled = np.logical_and(y_pred_ment_len_labelled,y_pred_prevalence_labelled).astype(int)
    get_and_display_results(y_test_labelled, y_pred_rules_labelled)
        
    print('both rule OR results - transferred')
    y_pred_rules_labelled = np.logical_or(y_pred_ment_len_labelled,y_pred_prevalence_labelled).astype(int)
    get_and_display_results(y_test_labelled, y_pred_rules_labelled)
    
    #for re-training
    for prevalance_th in ['0.005','0.0005','0.01']:
        print('prevalence rule results p %s' % prevalance_th)
        y_pred_prevalence_labelled = np.nan_to_num(df_filtered[['rule (prevalance th <= %s)' % prevalance_th]].to_numpy())
        get_and_display_results(y_test_labelled, y_pred_prevalence_labelled)
        
        y_pred_ment_len_labelled = y_pred_ment_len_labelled if prevalance_th == '0.005' else y_pred_ment_len_labelled_l4
        
        print('both rule AND results p %s' % prevalance_th)
        y_pred_rules_labelled = np.logical_and(y_pred_ment_len_labelled,y_pred_prevalence_labelled).astype(int)
        get_and_display_results(y_test_labelled, y_pred_rules_labelled)
        
        print('both rule OR results p %s' % prevalance_th)
        y_pred_rules_labelled = np.logical_or(y_pred_ment_len_labelled,y_pred_prevalence_labelled).astype(int)
        get_and_display_results(y_test_labelled, y_pred_rules_labelled)
    
    #bert-based contextual embedding results
    print('model non-masked results:')
    y_pred_test_nm_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_test_nm_labelled)
    
    if dataset == 'MIMIC-III':
        print('model non-masked ds results:')
        y_pred_test_nm_ds_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction ds']].to_numpy())
        get_and_display_results(y_test_labelled, y_pred_test_nm_ds_labelled)
    
    print('model non-masked fully supervised results:')
    y_pred_test_nm_sup_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction sup']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_test_nm_sup_labelled)
    
    if dataset == 'MIMIC-III':
        print('model non-masked ds fully supervised results:')
        y_pred_test_nm_ds_sup_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction ds sup']].to_numpy())
        get_and_display_results(y_test_labelled, y_pred_test_nm_ds_sup_labelled)
    
    print('model non-masked results (p0.005 l3):')
    y_pred_test_nm_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction p0.005 l3']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_test_nm_labelled)

    #best re-trained recall model and F1
    print('model non-masked results (p0.01 l4):')
    y_pred_test_nm_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction p0.01 l4']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_test_nm_labelled)
    
    #best re-trained F1 model
    print('model non-masked results (p0.0005 l4):')
    y_pred_test_nm_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction p0.0005 l4']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_test_nm_labelled)