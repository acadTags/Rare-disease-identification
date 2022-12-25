# analyse the semantic types of the concepts in ORDO2UMLS_ICD10_ICD9+titles_final_v2.xlsx from step0-om-ordo-umls-icd.
# input: (i)  ORDO2UMLS_ICD10_ICD9+titles_final_v2.xlsx
#        (ii) MRSTY-2020AB.RRF (i.e. MRSTY.RRF from UMLS2020AB)
# output:     ORDO2UMLS_ICD10_ICD9+titles_final_v3.xlsx - a updated ontology file that has semantic type column added

import pandas as pd
import re
from tqdm import tqdm

# Python program to illustrate union
# Without repetition
def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list

df = pd.read_excel("ORDO2UMLS_ICD10_ICD9+titles_final_v2.xlsx")

MRSTY_filename="MRSTY-2020AB.RRF"
df_sty = pd.read_csv(MRSTY_filename,
    delimiter='|',
    header=None,
    index_col=False,
    names=['CUI','STY','unknown1','unknown2','unknown3','unknown4','unknown5'],
    usecols=['CUI','STY'])

print(df.head())
print(df_sty.head())

#get dict of CUI to STYs from "MRSTY-2020AB.RRF"
dict_CUI_to_STY = df_sty.groupby('CUI')['STY'].apply(list).to_dict()

df['UMLS STY'] = ""
df['UMLS STY'] = df['UMLS STY'].apply(list)
pattern = "'(.*?)'"
list_STYs_all=[]
for i, row in tqdm(df.iterrows()):
    CUIs = re.findall(pattern,row['UMLS IDs'])
    CUIs = [CUI[len('UMLS:'):] for CUI in CUIs]
    list_STYs_all_from_CUI = [] # all unique STYs for the CUIs
    for CUI in CUIs:
        if CUI in dict_CUI_to_STY:
            list_STYs_from_CUI = dict_CUI_to_STY[CUI]
            list_STYs_all_from_CUI = Union(list_STYs_all_from_CUI,list_STYs_from_CUI)
        else:
            print(CUI, 'not in %s' % MRSTY_filename)    
    df.at[i,'UMLS STY'] = list_STYs_all_from_CUI

    list_STYs_all = list_STYs_all + [STY for STY in list_STYs_all_from_CUI if STY not in list_STYs_all]

print('all STYs from concepts:', list_STYs_all)

# get distribution
UMLS_STY_column_full_string = df['UMLS STY'].to_string()
for STY in list_STYs_all:
    STY_occ = UMLS_STY_column_full_string.count(STY)
    print(STY,STY_occ,float(STY_occ)/len(df))
    
n_non_STY=0
n=0
for i, row in tqdm(df.iterrows()):
    STYs = row['UMLS STY']
    if STYs == []:
        n_non_STY+=1
    n+=1
print('%.4f or %d/%d of concepts do not have STY.' % (float(n_non_STY)/n, n_non_STY,n))

df.to_excel('ORDO2UMLS_ICD10_ICD9+titles_final_v3.xlsx', index=False)

'''
2020AB
T047 3245 0.7984744094488189
T191 374 0.09202755905511811
T019 465 0.11441929133858268
T049 35 0.00861220472440945
T033 9 0.0022145669291338582
T046 19 0.004675196850393701
T190 13 0.0031988188976377952
T037 12 0.002952755905511811
T048 11 0.0027066929133858267
T020 3 0.0007381889763779527
T184 1 0.00024606299212598425
4064it [00:00, 70478.51it/s]
0.0091 or 37/4064 of concepts do not have STY.

2020AA
T047 3244 0.7982283464566929
T191 374 0.09202755905511811
T019 470 0.1156496062992126
T049 35 0.00861220472440945
T033 11 0.0027066929133858267
T046 18 0.0044291338582677165
T190 14 0.0034448818897637795
T037 12 0.002952755905511811
T048 11 0.0027066929133858267
T020 2 0.0004921259842519685
T184 1 0.00024606299212598425
4064it [00:00, 70621.89it/s]
0.0089 or 36/4064 of concepts do not have STY.

2019AB
T047 3249 0.7994586614173228
T191 373 0.09178149606299213
T019 473 0.11638779527559055
T049 34 0.008366141732283465
T033 11 0.0027066929133858267
T046 19 0.004675196850393701
T190 14 0.0034448818897637795
T037 12 0.002952755905511811
T048 11 0.0027066929133858267
T020 2 0.0004921259842519685
T184 1 0.00024606299212598425
4064it [00:00, 69154.36it/s]
0.0079 or 32/4064 of concepts do not have STY.
'''