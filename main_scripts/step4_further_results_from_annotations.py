# this program calculates the binary filtering of UMLS concept extraction from the 'for validation - SemEHR ori.xlsx' sheet.

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
    data_sheet_fn = 'data annotation/raw annotations (with model predictions)/for validation - SemEHR ori (MIMIC-III-DS, free text removed, with predictions).xlsx'
    df = pd.read_excel(data_sheet_fn,engine='openpyxl')
    final_ann_available = True
    
    # 2. filtering the data to evaluate - uncomment some of the options below
    
    # get results of rule-labelled ones which are also manually labelled
    #df = df[:400] # validation: only evaluate the first k (as 400) data in this category
    df = df[-673:] # testing
    
    # positive only, rule-labelled 
    #df_filtered = df[~ df['pos label: both rules applied'].isna()]# & (~ df['manual label from ann1'].isna())]
    
    # negative only, rule-labelled
    #df_filtered = df[~ df['neg label: only when both rule 0'].isna()]# & (~ df['manual label from ann1'].isna())]
    
    # all positive + negative, rule-labelled, all seen
    #df_filtered = df[((~ df['neg label: only when both rule 0'].isna()) | (~ df['pos label: both rules applied'].isna()))]# & (~ df['manual label from ann1'].isna())]
    
    # get results of unseen ones
    # only mention length rule applied, unseen
    #df_filtered = df[(df['rule (mention length >3)']==1) & (df['rule (prevalance th <= 0.005)']==0)]# & (~ df['manual label from ann1'].isna())]
    
    # only prevalence rule applied, unseen
    #df_filtered = df[(df['rule (mention length >3)']==0) & (df['rule (prevalance th <= 0.005)']==1)]# & (~ df['manual label from ann1'].isna())]
    
    # any of the two rules applied, unseen
    #df_filtered = df[(df['neg label: only when both rule 0'].isna()) & (df['pos label: both rules applied'].isna())]# & (~ df['manual label from ann1'].isna())]
    
    # get results of the first or last k data instances
    #df_filtered = df[:250]
    #df_filtered = df[:200]
    #df_filtered = df[-673:]
    
    # no filtering
    df_filtered = df
    
    print(len(df_filtered))#,df_filtered)
    
    # 3. get gold results from the sheet
    if final_ann_available:
        y_test_labelled = df_filtered[['gold text-to-UMLS label']].to_numpy()
    else:
        y_test_labelled = df_filtered[['manual label from ann1']].to_numpy()
    y_test_labelled = np.where((y_test_labelled==-1) | (y_test_labelled == np.nan), 0, y_test_labelled) # 0 when not applicable or not filled
     
    print(y_test_labelled.shape)
    #print(y_test_labelled)
    #sys.exit(0)
    
    # 4. get prediction results and calculate precision, recall, and F1
    # also export the results to a csv file
    
    #print metrics - unappplicable or nan value in the predictions are changed to 0 before the calculation
    print('rule-based weakly annotation results - all')
    y_rule_based_weak_labelled = np.nan_to_num(df_filtered[['pos label: both rules applied']].to_numpy())
    #simply change nans in the pos label column to 0, as neg column are all 0 or nans.
    get_and_display_results(y_test_labelled, y_rule_based_weak_labelled)
    
    for tolerance in [0,5,10,15]:
        g_api_result_col_name = 'Google Healthcare API cw5%s' % (' t' + str(tolerance) if tolerance != 0 else ' new')
        print(g_api_result_col_name) # google API results queried on 18 March 2021
        y_pred_Google_API_test_labelled = df_filtered[[g_api_result_col_name]].to_numpy() 
        get_and_display_results(y_test_labelled, y_pred_Google_API_test_labelled)
    
    #tune the acc_threshold and window_size for the MedCAT tool
    for acc_threshold in np.arange(0,0.3,0.1):
        for window_size in [5,10]:
            for model_size in ['small','medium','large']:
                for metaAnn_mark in ['',' metaAnn']:
                    for tolerance in [0,5,10]:
                        medcat_result_col_name = 'MedCAT cw%d %s%s%s' % (window_size,model_size,' t' + str(tolerance) if tolerance != 0 else '',metaAnn_mark)
                        if medcat_result_col_name in df_filtered:
                            print('MedCAT results cw%d %s%s%s acc_th%.1f' % (window_size,model_size,' t' + str(tolerance) if tolerance != 0 else '',metaAnn_mark,acc_threshold))
                            y_pred_MedCAT_test_labelled = df_filtered[[medcat_result_col_name]].to_numpy() # get the 
                            y_pred_MedCAT_test_labelled_bin = y_pred_MedCAT_test_labelled >= acc_threshold
                            get_and_display_results(y_test_labelled, y_pred_MedCAT_test_labelled_bin)
    
    # if len(df_filtered) <= 200:  # google API results queried in Nov 2020, only 200 data were queried by the time
        # print('Google Healthcare API results cw5')
        # y_pred_Google_API_test_labelled = df_filtered[['Google Healthcare API cw5']].to_numpy() 
        # get_and_display_results(y_test_labelled, y_pred_Google_API_test_labelled)
    
        # print('Google Healthcare API results cw10')
        # y_pred_Google_API_test_labelled = df_filtered[['Google Healthcare API cw10']].to_numpy() 
        # get_and_display_results(y_test_labelled, y_pred_Google_API_test_labelled)
        
        # print('Google Healthcare API results cw50')
        # y_pred_Google_API_test_labelled = df_filtered[['Google Healthcare API cw50']].to_numpy() 
        # get_and_display_results(y_test_labelled, y_pred_Google_API_test_labelled)
    
    print('SemEHR results')
    y_pred_SemEHR_test_labelled = df_filtered[['SemEHR label']].to_numpy() 
    get_and_display_results(y_test_labelled, y_pred_SemEHR_test_labelled)
    
    print('mention length rule results')
    y_pred_ment_len_labelled = df_filtered[['rule (mention length >3)']].to_numpy()
    get_and_display_results(y_test_labelled, y_pred_ment_len_labelled)
    
    print('prevalence rule results')
    y_pred_prevalence_labelled = df_filtered[['rule (prevalance th <= 0.005)']].to_numpy()
    get_and_display_results(y_test_labelled, y_pred_prevalence_labelled)
    
    print('both rule AND results')
    y_pred_rules_labelled = np.logical_and(y_pred_ment_len_labelled,y_pred_prevalence_labelled).astype(int)
    get_and_display_results(y_test_labelled, y_pred_rules_labelled)
    
    print('both rule OR results')
    y_pred_rules_labelled = np.logical_or(y_pred_ment_len_labelled,y_pred_prevalence_labelled).astype(int)
    get_and_display_results(y_test_labelled, y_pred_rules_labelled)
    
    #averaged word embedding results
    for w2v_model_name in ['w2v_caml_100','w2v_mimic_300','w2v_mimic_768']:
        for masked_training in [False,True]:
            for use_doc_struc in [False,True]:
                w2v_res_column_name = 'model %s prediction%s%s' % (w2v_model_name, ' (masked training)' if masked_training else '', ' ds' if use_doc_struc else '')
                if w2v_res_column_name in df_filtered:
                    print('model ave w2v emb results %s %s%s' % (w2v_model_name,'masked' if masked_training else 'non-masked',' ds' if use_doc_struc else ''))
                    y_pred_ave_w2v_test_labelled = np.nan_to_num(df_filtered[[w2v_res_column_name]].to_numpy()) # get the 
                    get_and_display_results(y_test_labelled, y_pred_ave_w2v_test_labelled)
        
    #bert-based contextual embedding results
    print('model non-masked results:')
    y_pred_test_nm_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_test_nm_labelled)
    
    print('model non-masked hf fine-tune results:')
    y_pred_test_nm_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm hf prediction']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_test_nm_labelled)
    
    for model_name in ['blueBERTnorm','pubmedBERT','BERTbase', 'SapBERT']:
        for window_size in [5,10,20]:
            lm_res_column_name = 'model %s prediction ds%s' % (model_name,' cw' + str(window_size) if window_size != 5 else '')
            if lm_res_column_name in df_filtered:
                print('model %s non-masked ds%s results:' % (model_name,' cw' + str(window_size) if window_size != 5 else ''))    
                y_pred_test_nm_ds_labelled = np.nan_to_num(df_filtered[[lm_res_column_name]].to_numpy())
                get_and_display_results(y_test_labelled, y_pred_test_nm_ds_labelled)
    
    print('model non-masked ds hf fine-tune results:') # used huggingface transformers to finetune
    y_pred_test_nm_ds_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm hf prediction ds']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_test_nm_ds_labelled)
    
    print('model non-masked ds data sel results:')
    y_pred_test_nm_ds_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction ds tr9000']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_test_nm_ds_labelled)
    
    print('model blueBERTlarge non-masked ds results:')
    y_pred_test_nm_ds_large_labelled = np.nan_to_num(df_filtered[['model blueBERTlarge prediction ds']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_test_nm_ds_large_labelled)
    
    print('model masked results:')
    y_pred_test_m_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction (masked training)']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_test_m_labelled)
    
    print('model masked hf fine-tune results:')
    y_pred_test_m_hf_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm hf prediction (masked training)']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_test_m_hf_labelled)
    
    print('model masked ds results:')
    y_pred_test_m_ds_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm prediction (masked training) ds']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_test_m_ds_labelled)
    
    print('model masked ds hf fine-tune results:') # used huggingface transformers to finetune
    y_pred_test_m_ds_labelled = np.nan_to_num(df_filtered[['model blueBERTnorm hf prediction (masked training) ds']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_test_m_ds_labelled)
    
    print('model blueBERTlarge masked ds results:')
    y_pred_test_m_ds_large_labelled = np.nan_to_num(df_filtered[['model blueBERTlarge prediction (masked training) ds']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_test_m_ds_large_labelled)
    
    # rule-based model ensembling results
    print('rule-based model ensemble best scenario results:')
    #y_pred_test_m_labelled_ensemb = np.logical_or(y_pred_test_m_labelled,y_pred_test_m_ds_large_labelled).astype(int)
    y_pred_rule_based_model_ensemble = rule_based_model_ensemble(y_pred_ment_len_labelled, y_pred_prevalence_labelled, y_pred_test_nm_ds_labelled, y_pred_test_m_labelled)
    get_and_display_results(y_test_labelled, y_pred_rule_based_model_ensemble)
    # add the results to a column if not there
    if 'model ensemble best scenario' not in df_filtered.columns:
        #print(y_pred_rule_based_model_ensemble.shape)
        df_filtered['model ensemble best scenario']=pd.Series(np.squeeze(y_pred_rule_based_model_ensemble,axis=1)) # squeeze to one dimension
    
    print('rule-based model ensemble blueBERTnorm results:')
    y_pred_rule_based_model_ensemble = rule_based_model_ensemble(y_pred_ment_len_labelled, y_pred_prevalence_labelled, y_pred_test_nm_labelled, y_pred_test_m_labelled)
    get_and_display_results(y_test_labelled, y_pred_rule_based_model_ensemble)
    
    # print('rule-based model ensemble blueBERTlarge results:')
    # y_pred_rule_based_model_ensemble = rule_based_model_ensemble(y_pred_ment_len_labelled, y_pred_prevalence_labelled, y_pred_test_nm_labelled, y_pred_test_m_labelled)
    # get_and_display_results(y_test_labelled, y_pred_rule_based_model_ensemble)
    
    print('rule-based model ensemble ds blueBERTnorm results:')
    y_pred_rule_based_model_ensemble = rule_based_model_ensemble(y_pred_ment_len_labelled, y_pred_prevalence_labelled, y_pred_test_nm_ds_labelled, y_pred_test_m_ds_labelled)
    get_and_display_results(y_test_labelled, y_pred_rule_based_model_ensemble)
    
    print('rule-based model ensemble ds blueBERTlarge results:')
    y_pred_rule_based_model_ensemble = rule_based_model_ensemble(y_pred_ment_len_labelled, y_pred_prevalence_labelled, y_pred_test_nm_ds_labelled, y_pred_test_m_ds_large_labelled)
    get_and_display_results(y_test_labelled, y_pred_rule_based_model_ensemble)
    
    print('model non-masked ds, fully supervised with 400 validation data:')
    y_pred_test_nm_ds_labelled_full_sup = np.nan_to_num(df_filtered[['full supervised model blueBERTnorm prediction ds']].to_numpy())
    get_and_display_results(y_test_labelled, y_pred_test_nm_ds_labelled_full_sup)
    
    # to export the filtered df file
    #df_filtered.to_excel('for validation - SemEHR ori - results updated.xlsx',index=False)
    #df_filtered.to_excel('for validation - SemEHR ori - rule-labelled 135 manual results.xlsx',index=False)