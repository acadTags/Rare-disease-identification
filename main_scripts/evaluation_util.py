from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# input gold labels and pred labels
# display prec,rec,F1 and confusion matrix
# also return the prec,rec,F1
def get_and_display_results(y_true, y_pred, report_confusion_mat = False):
    prec, rec, f1 = precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print('test precision: %s test recall: %s test F1: %s' % (str(prec), str(rec), str(f1)))
    print('tp %s tn %s fp %s fn %s\n' % (str(tp),str(tn),str(fp),str(fn)))
    if not report_confusion_mat:
        return prec, rec, f1
    else:    
        return prec, rec, f1, tn, fp, fn, tp

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