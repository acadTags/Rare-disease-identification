from rare_disease_id_util import isNotGroupOfDisorders, isDisease
import pandas as pd
from tqdm import tqdm

df = pd.read_excel('for validation - row id added - by text.xlsx')

dict_ordo_isNotGroupOfDisorders = {}
dict_ordo_idDisease = {}

df['ORDOisNotGroupOfDisorder'] = ""
df['ORDOisDisease'] = ""

for i, row in tqdm(df.iterrows()):
    ORDO_ID = row['ORDO with desc']
    ORDO_ID = ORDO_ID[:ORDO_ID.find(' ')]
    if dict_ordo_isNotGroupOfDisorders.get(ORDO_ID,None) == None:
        isORDONotGroupOfDisorders = isNotGroupOfDisorders(ORDO_ID)
        dict_ordo_isNotGroupOfDisorders[ORDO_ID] = isORDONotGroupOfDisorders
        row['ORDOisNotGroupOfDisorder'] = isORDONotGroupOfDisorders
        
        isORDODisease = isDisease(ORDO_ID)
        dict_ordo_idDisease[ORDO_ID] = isORDODisease
        row['ORDOisDisease'] = isORDODisease
    else:    
        row['ORDOisNotGroupOfDisorder'] = dict_ordo_isNotGroupOfDisorders[ORDO_ID]
        row['ORDOisDisease'] = dict_ordo_idDisease[ORDO_ID]

df.to_excel('for validation - row id added - by text - onto filtered.xlsx',index=False)
