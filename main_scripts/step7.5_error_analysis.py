#error analysis for ORDO concept identification results

import numpy as np
import pandas as pd

#df = pd.read_excel('for validation - SemEHR ori.xlsx')
df = pd.read_excel('data annotation/raw annotations (with model predictions)/for validation - SemEHR ori (MIMIC-III-DS, free text removed, with predictions).xlsx',engine='openpyxl')

export_df_for_analyses = True

# a non-masked model to evaluate
model_column_name = 'model blueBERTnorm prediction ds tr9000'
#model_column_name = 'model blueBERTnorm prediction ds'
#model_column_name = 'model blueBERTnorm prediction'

# the masked model
model_column_name_masked = 'model blueBERTnorm prediction (masked training)'
model_column_name_ORDO = model_column_name + ' ORDO'
UMLS_to_ORDO_rule_matching_cl_name = 'ORDOisNotGroupOfDisorder'
gold_text_ORDO_cl_name = 'gold text-to-ORDO label'
gold_UMLS_ORDO_cl_name = 'gold UMLS-to-ORDO label'
gold_text_UMLS_cl_name = 'gold text-to-UMLS label'

#false positive: fp, predicted 1 but gold as 0
df_fp = df[(df[model_column_name_ORDO]==1) & (df[gold_text_ORDO_cl_name]==0)]

#   fp due to text-to-UMLS matching
df_fp_tU = df_fp[(df_fp[gold_text_UMLS_cl_name] == 0) & (df_fp[model_column_name] == 1)] # predicted 1 but gold as 0
#       due to weak supervision (labelled positive)         
df_fp_tU_WS = df_fp_tU[df_fp_tU['pos label: both rules applied'] == 1]
#           but addressed by masked encoding 
df_fp_tU_WS_ME = df_fp_tU_WS[df_fp_tU_WS[model_column_name_masked]==0]
#       not due to weak supervision
df_fp_tU_notWS = df_fp_tU[df_fp_tU['pos label: both rules applied'] != 1]
#           but addressed by masked encoding 
df_fp_tU_notWS_ME = df_fp_tU_notWS[df_fp_tU_notWS[model_column_name_masked]==0]
#       in overall can be address by masked encoding
df_fp_tU_ME = df_fp_tU[df_fp_tU[model_column_name_masked] == 0]

# fp due to onto matching: UMLS-to-ORDO matching
df_fp_UO = df_fp[(df_fp[gold_UMLS_ORDO_cl_name] == 0) & (df_fp[UMLS_to_ORDO_rule_matching_cl_name] == 1)] # predicted 1 but gold as 0
df_fp_UO_unique = df_fp_UO.drop_duplicates(subset = ['UMLS with desc', 'ORDO with desc'], keep='first')

# fp due to both text-to-UMLS matching and UMLS-to-ORDO matching
df_fp_UO_tU = df_fp_UO[(df_fp_UO[gold_text_UMLS_cl_name]==0) & (df_fp_UO[model_column_name] == 1)]

#false negative: fn, predicted 0 but gold as 1
df_fn = df[(df[model_column_name_ORDO]==0) & (df[gold_text_ORDO_cl_name]==1)]

#   fn due to text-to-UMLS matching
df_fn_tU = df_fn[(df_fn[gold_text_UMLS_cl_name] == 1) & (df_fn[model_column_name] == 0)] # predict 0 but gold as 1
#       due to weak supervision (labelled negative)         
df_fn_tU_WS = df_fn_tU[df_fn_tU['neg label: only when both rule 0'] == 0]
#           but addressed by masked encoding 
df_fn_tU_WS_ME = df_fn_tU_WS[df_fn_tU_WS[model_column_name_masked]==1]
#       not due to weak supervision
df_fn_tU_notWS = df_fn_tU[df_fn_tU['neg label: only when both rule 0'] != 0]
#           but addressed by masked encoding 
df_fn_tU_notWS_ME = df_fn_tU_notWS[df_fn_tU_notWS[model_column_name_masked]==1]
#       in overall can be address by masked encoding
df_fn_tU_ME = df_fn_tU[df_fn_tU[model_column_name_masked] == 1]

#   fn due to onto matching: UMLS-to-ORDO matching
df_fn_UO = df_fn[(df_fn[gold_UMLS_ORDO_cl_name] == 1) & (df_fn[UMLS_to_ORDO_rule_matching_cl_name] == 0)] # predicted 0 but gold as 1
df_fn_UO_unique = df_fn_UO.drop_duplicates(subset = ['UMLS with desc', 'ORDO with desc'], keep='first')

#   fn due to both text-to-UMLS matching and UMLS-to-ORDO matching
df_fn_UO_tU = df_fn_UO[(df_fn_UO[gold_text_UMLS_cl_name]==1) & (df_fn_UO[model_column_name] == 0)]

#overall: fp+fn

df_fp_fn_UO = pd.concat([df_fp_UO,df_fn_UO],ignore_index=True)
df_fp_fn_UO_unique = df_fp_fn_UO.drop_duplicates(subset = ['UMLS with desc', 'ORDO with desc'], keep='first')

print('Error Analysis Report')

print('fp:%s fn:%s' % (str(len(df_fp)), str(len(df_fn))))
print('  fp due to text-UMLS matching: %s (or %.2f%%)' % (len(df_fp_tU),100*len(df_fp_tU)/len(df_fp)))
print('    those due to weak supervsion: %s (or %.2f%%)' % (len(df_fp_tU_WS),100*len(df_fp_tU_WS)/len(df_fp_tU)))
print('      but can be addressed by masked encoding: %s (or %.2f%%)' % (len(df_fp_tU_WS_ME),100*len(df_fp_tU_WS_ME)/len(df_fp_tU_WS)))
print('    those not due to weak supervsion: %s (or %.2f%%)' % (len(df_fp_tU_notWS),100*len(df_fp_tU_notWS)/len(df_fp_tU)))
print('      but can be addressed by masked encoding: %s (or %.2f%%)' % (len(df_fp_tU_notWS_ME),100*len(df_fp_tU_notWS_ME)/len(df_fp_tU_notWS)))
print('    those in overall can be addressed by masked encoding: %s (or %.2f%%)' % (len(df_fp_tU_ME),100*len(df_fp_tU_ME)/len(df_fp_tU)))

print('  fp due to UMLS-ORDO matching: %s (or %.2f%%)' % (len(df_fp_UO),100*len(df_fp_UO)/len(df_fp)))
print('     the unique ones: %s' % (len(df_fp_UO_unique)))
print('  fp due to both t-U and U-O matching: %s (or %.2f%%)' % (len(df_fp_UO_tU),100*len(df_fp_UO_tU)/len(df_fp)))
print('')

print('  fn due to text-UMLS matching: %s (or %.2f%%)' % (len(df_fn_tU),100*len(df_fn_tU)/len(df_fn)))
print('    those due to weak supervsion: %s (or %.2f%%)' % (len(df_fn_tU_WS),100*len(df_fn_tU_WS)/len(df_fn_tU)))
print('      but can be addressed by masked encoding: %s (or %.2f%%)' % (len(df_fn_tU_WS_ME),100*len(df_fn_tU_WS_ME)/len(df_fn_tU_WS)))
print('    those not due to weak supervsion: %s (or %.2f%%)' % (len(df_fn_tU_notWS),100*len(df_fn_tU_notWS)/len(df_fn_tU)))
print('      but can be addressed by masked encoding: %s (or %.2f%%)' % (len(df_fn_tU_notWS_ME),100*len(df_fn_tU_notWS_ME)/len(df_fn_tU_notWS)))
print('    those in overall can be addressed by masked encoding: %s (or %.2f%%)' % (len(df_fn_tU_ME),100*len(df_fn_tU_ME)/len(df_fn_tU)))

print('  fn due to UMLS-ORDO matching: %s (or %.2f%%)' % (len(df_fn_UO),100*len(df_fn_UO)/len(df_fn)))
print('     the unique ones: %s' % (len(df_fn_UO_unique)))
print('  fn due to both t-U and U-O matching: %s (or %.2f%%)' % (len(df_fn_UO_tU),100*len(df_fn_UO_tU)/len(df_fn)))
print('')

print('  fp+fn in overall due to text-UMLS matching: %s (or %.2f%%)' % (len(df_fp_tU) + len(df_fn_tU),100*(len(df_fp_tU) + len(df_fn_tU))/(len(df_fp) +len(df_fn))))
print('  fp+fn in t-U overall due to weak supervision: %s (or %.2f%%)' % (len(df_fp_tU_WS) + len(df_fn_tU_WS),100*(len(df_fp_tU_WS) + len(df_fn_tU_WS))/(len(df_fp_tU) +len(df_fn_tU))))
print('  fp+fn in t-U overall can be addressed by masked encoding: %s (or %.2f%%)' % (len(df_fp_tU_ME) + len(df_fn_tU_ME),100*(len(df_fp_tU_ME) + len(df_fn_tU_ME))/(len(df_fp_tU) +len(df_fn_tU))))
print('  fp+fn in overall due to UMLS-ORDO matching: %s (or %.2f%%)' % (len(df_fp_UO) + len(df_fn_UO),100*(len(df_fp_UO) + len(df_fn_UO))/(len(df_fp) +len(df_fn))))
print('     the unique ones: %s' % (len(df_fp_fn_UO_unique)))
print('  fp+fn in overall due to both t-U and U-O matching: %s (or %.2f%%)' % (len(df_fp_UO_tU) + len(df_fn_UO_tU),100*(len(df_fp_UO_tU) + len(df_fn_UO_tU))/(len(df_fp) +len(df_fn))))

if export_df_for_analyses:
    df_fp_tU_notWS.to_excel('error_analysis_fp_tU_notWS_%s.xlsx' % model_column_name,index=False)
    df_fp_UO_unique.to_excel('error_analysis_fp_UO_unique_%s.xlsx' % model_column_name,index=False)
    
    df_fn_tU_notWS.to_excel('error_analysis_fn_tU_notWS_%s.xlsx' % model_column_name,index=False)
    df_fn_UO_unique.to_excel('error_analysis_fn_UO_unique_%s.xlsx' % model_column_name,index=False)