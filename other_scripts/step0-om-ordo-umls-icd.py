#get icd9 from ORDO-UMLS-ICD10 pairs, also get the descriptions of icd9.
#through: (i) ICD10-ICD9 MoH, New zealand, masterb10.xls, https://www.health.govt.nz/nz-health-statistics/data-references/mapping-tools/mapping-between-icd-10-and-icd-9
#         (ii) UMLS-ICD9 from bioportal, https://bioportal.bioontology.org/ontologies/ICD9CM

#code adapted from minhong's icd102icd9.py

#input ordo-umls-icd file previously created, 'ORDO2UMLS_ICD.xlsx', based on ORDO ver3.0 (released 07/03/2020) from https://bioportal.bioontology.org/ontologies/ORDO
#output matched icd9s with the two ontology matching paths
          
import pandas as pd
import re
from rare_disease_id_util import ordo2icd10FromJSON, umls2icd9List_bp, union
from tqdm import tqdm

pd.set_option("display.max_colwidth", 10000) # allowing showing and matching to long sequence

# import data
df1 = pd.read_excel("ORDO2UMLS_ICD.xlsx", sheet_name="UMLS and ICD")
#df2 = pd.read_excel("ORDO2UMLS_ICD.xlsx", sheet_name="UMLS")
#df3 = pd.read_excel("ORDO2UMLS_ICD.xlsx", sheet_name="ICD")

# get icd10 codes - those directly from the csv file and those with exact or narrower matching from JSON API by ORDO
df2 = df1.assign()#[:100]
df2['icd10_all_from_csv'] = ''
df2['icd10_all_from_csv'] = df2['icd10_all_from_csv'].apply(list)
# df2['icd10_all_from_json'] = ''
# df2['icd10_all_from_json'] = df2['icd10_all_from_json'].apply(list)
df2['icd10_exact_or_narrower_from_json'] = ''
df2['icd10_exact_or_narrower_from_json'] = df2['icd10_exact_or_narrower_from_json'].apply(list)
pattern = "'(.*?)'"
for i, row in tqdm(df2.iterrows()):
    icd10s = re.findall(pattern,row['ICD IDs'])
    icd10s = [icd10[7:].replace('.','') for icd10 in icd10s]
    df2.at[i, 'icd10_all_from_csv'] = icd10s
    
    ORDO_ID_url = row['ORDO ID']
    # icd10s_all_from_json = ordo2icd10FromJSON(ORDO_ID_url,exact_or_narrower_only=False)
    # icd10s_all_from_json = [icd10.replace('.','') for icd10 in icd10s_all_from_json]
    # df2.at[i, 'icd10_all_from_json'] = icd10s_all_from_json
    
    icd10s_E_N_from_json = ordo2icd10FromJSON(ORDO_ID_url,exact_or_narrower_only=True)
    icd10s_E_N_from_json = [icd10.replace('.','') for icd10 in icd10s_E_N_from_json]
    df2.at[i, 'icd10_exact_or_narrower_from_json'] = icd10s_E_N_from_json
    
# import the map, from health.govt.nz
map = pd.read_excel('masterb10.xls')

# map icd10_all_from_csv/icd10_exact_or_narrower_from_json -> icd9, using MoH, New zealand mapping
df2['icd9-NZ'] = ""
df2['icd9-NZ'] = df2['icd9-NZ'].apply(list)
df2['icd9-NZ-E-N'] = ""
df2['icd9-NZ-E-N'] = df2['icd9-NZ-E-N'].apply(list)
for i, row in tqdm(df2.iterrows()):
    icd10s_all_csv = row['icd10_all_from_csv']
    for j,k in enumerate(icd10s_all_csv):
        icd10_tmp = k
        #tabletype_tmp = map[map['ICD10']==icd10_tmp]['TABLETYP'].to_string(index=False)
        #print(map['ICD10']==icd10_tmp)
        num_tmp = map[map['ICD10']==icd10_tmp]['Pure Victorian Logical'].to_string(index=False)
        #icd9_tmp = tabletype_tmp + num_tmp
        icd9_tmp = num_tmp
        #print(icd9_tmp)
        if icd9_tmp != 'Series([], )':
            df2.at[i,'icd9-NZ'].append(icd9_tmp)
    
    icd10s_E_N_json = row['icd10_exact_or_narrower_from_json']
    for j,k in enumerate(icd10s_E_N_json):
        icd10_tmp = k
        num_tmp = map[map['ICD10']==icd10_tmp]['Pure Victorian Logical'].to_string(index=False)
        icd9_tmp = num_tmp
        if icd9_tmp != 'Series([], )':
            df2.at[i,'icd9-NZ-E-N'].append(icd9_tmp)
            
# import the map_icd, from MIMIC-III D_ICD_DIAGNOSES.csv
map_icd = pd.read_csv('D_ICD_DIAGNOSES.csv')

df2['icd9-short-titles'] = ""
df2['icd9-short-titles'] = df2['icd9-short-titles'].apply(list)
df2['icd9-long-titles'] = ""
df2['icd9-long-titles'] = df2['icd9-long-titles'].apply(list)
df2['icd9-short-titles-E-N'] = ""
df2['icd9-short-titles-E-N'] = df2['icd9-short-titles-E-N'].apply(list)
df2['icd9-long-titles-E-N'] = ""
df2['icd9-long-titles-E-N'] = df2['icd9-long-titles-E-N'].apply(list)
for i, row in tqdm(df2.iterrows()):
    icd9s = row['icd9-NZ']
    for j,k in enumerate(icd9s):
        icd9_tmp = k.strip()
        short_title_tmp = map_icd[map_icd['ICD9_CODE']==icd9_tmp]['SHORT_TITLE'].to_string(index=False)
        long_title_tmp = map_icd[map_icd['ICD9_CODE']==icd9_tmp]['LONG_TITLE'].to_string(index=False)
        if short_title_tmp != 'Series([], )':
            df2.at[i,'icd9-short-titles'].append(short_title_tmp)
        if long_title_tmp != 'Series([], )':    
            df2.at[i,'icd9-long-titles'].append(long_title_tmp)
    
    icd9s = row['icd9-NZ-E-N']
    for j,k in enumerate(icd9s):
        icd9_tmp = k.strip()
        short_title_tmp = map_icd[map_icd['ICD9_CODE']==icd9_tmp]['SHORT_TITLE'].to_string(index=False)
        long_title_tmp = map_icd[map_icd['ICD9_CODE']==icd9_tmp]['LONG_TITLE'].to_string(index=False)
        if short_title_tmp != 'Series([], )':
            df2.at[i,'icd9-short-titles-E-N'].append(short_title_tmp)
        if long_title_tmp != 'Series([], )':    
            df2.at[i,'icd9-long-titles-E-N'].append(long_title_tmp)
            
# map umls-icd9, using bioportal mapping
df2['icd9-bp'] = ""
df2['icd9-pref-label-bp'] = ""
map=None #initialise map for mapping
for i, row in tqdm(df2.iterrows()):
    UMLS_IDs = re.findall(pattern,row['UMLS IDs'])
    list_icd9, list_icd9_pref_label = [], []
    for umls_id in UMLS_IDs:
        umls_id = umls_id[5:]
        list_icd9_tmp,list_icd9_pref_label_tmp,map = umls2icd9List_bp(umls_id,map=map)
        #print(map)
        list_icd9 = list_icd9 + [icd9_code for icd9_code in list_icd9_tmp if icd9_code not in list_icd9] #union(list_icd9,list_icd9_tmp)    
        list_icd9_pref_label = list_icd9_pref_label + [icd9_pref_label for icd9_pref_label in list_icd9_pref_label_tmp if icd9_pref_label not in list_icd9_pref_label] #union(list_icd9_pref_label,list_icd9_pref_label_tmp)
    df2.at[i, 'icd9-bp'] = list_icd9
    df2.at[i, 'icd9-pref-label-bp'] = list_icd9_pref_label
    
# output the results
df2.to_excel('ORDO2UMLS_ICD10_ICD9+titles_final_v2.xlsx', index=False)


