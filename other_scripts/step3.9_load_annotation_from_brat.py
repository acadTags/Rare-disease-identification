# load the annotations from Brat
# brat files were generated using generate_brat_annotations.py
# then domain experts annotate the brat files by ticking or unticking the true_phenotype/false_phenotype, and may add notes if necessary

# this program then post-process the brat annotations and fill the results to the excel annotation sheet, for validation - SemEHR ori.xlsx
# this is a step to form the final gold standard annotation.

import pandas as pd
import numpy as np
import os 
from tqdm import tqdm

annotator = 'ann2' 
annotator_first_name = 'ann2'
annotation_column_name = 'manual label from %s' % annotator_first_name
ann_notes_column_name = 'Notes from %s' % annotator_first_name

#ann_file_folder_path = './brat_ann_rd_%s/' % annotator
#ann_file_folder_path = './brat_radiology_rd_%s/' % annotator
ann_file_folder_path = './tayside-raredisease/%s/' % annotator_first_name

#validation_data_sheet_fn = 'for validation - SemEHR ori.xlsx'
#validation_data_sheet_fn = 'for validation - 1000 docs - ori - MIMIC-III-rad.xlsx'
validation_data_sheet_fn = 'for validation - 5000 docs - ori - tayside - rad - ori.xlsx'
df = pd.read_excel(validation_data_sheet_fn)
print('number of rows of data:',len(df))

#for Tayside: filter out the manually added rows from human annotation
if 'manually added data' in df.columns:
    print(True)
    df = df[df['manually added data']!=True]
    print('number of rows of data with manually added rows:',len(df))

#the two lines below only used when extracting an annotators' results for the first time (i.e. with a new float type column)
df[annotation_column_name] = ''
df[ann_notes_column_name] = ''
for ind, (i, row) in tqdm(enumerate(df.iterrows())):
    # construct brat annotation file name
    brat_ann_file_id = str(ind+1)
    # add 0s if only less than 4 digits in the id
    brat_ann_file_id = '0'*(4-len(brat_ann_file_id)) + brat_ann_file_id if len(brat_ann_file_id) <= 3 else brat_ann_file_id
    doc_row_id = str(row['doc row ID'])
    if doc_row_id[0] == '[' and doc_row_id[-1] == ']':
        doc_row_id = doc_row_id[1:-1] #remove the [] sign if 
    doc_struc = row['document structure']
    brat_ann_file_name = '%s-doc-%s%s.ann' % (brat_ann_file_id,doc_row_id, '-' + doc_struc if not pd.isna(doc_struc) else '')
    brat_ann_file_name = os.path.join(ann_file_folder_path,brat_ann_file_name)
    
    try:
        with open(brat_ann_file_name,encoding='utf-8') as f_content:
            list_content = f_content.readlines()
        list_content = [x.strip() for x in list_content]
        content = '\n'.join(list_content)
        
        ann = -1
        ann_notes = ''
        # 0: false_phenotype
        # 1: true_phenotype
        # -1: unsure or not annotated
        # -2: true_phenotype but negative context (this should not exist)
        for ann_row in list_content[1:]:                
            # if false_phenotype then the annotation is 0
            if 'false_phenotype' in ann_row:
                ann = 0                
            # if true_phenotype then we check whether the context annotation is positive, if not, set as -2 (this should not happen as all true_phenotype cases should have a positive context)
            elif 'true_phenotype' in ann_row:
                if '\tpositive ' in content:
                    ann = 1    
                else:
                    ann = -2
                    #print(str(ann),list_content) 
            # if unsure then we set the annotation as -1
            elif '\tunsure ' in ann_row:
                # add the inclined annotation (if annoated as well) to the annotaton notes
                if 'true_phenotype' in content:
                    ann_notes = ann_notes + 'unsure but tends to be true_phenotype'
                if 'false_phenotype' in content:
                    ann_notes = ann_notes + 'unsure but tends to be false_phenotype'                    
                ann = -1
                
        for ann_row in list_content[1:]:
            if '\tAnnotatorNotes' in ann_row:
                ann_notes = ann_notes + ' ' + ann_row.split('\t')[-1]
        
        #if annotator == 'MW' or annotator == 'HD' and not '\tpositive ' in content:
        if not '\tpositive ' in content:
            ann_notes = '(context:%s)' % 'negative' + ' ' + ann_notes.strip()
        #print(str(ann),list_content,ann_notes)
        #print(content)
        
        # set the retrieved values to the dataframe
        #if np.isnan(df.at[i,annotation_column_name]):
        df.at[i,annotation_column_name] = ann
        df.at[i,ann_notes_column_name] = ann_notes.strip()
    except FileNotFoundError:
        print(brat_ann_file_name, 'not found')
    #print(i, df.at[i,annotation_column_name])

df.to_excel(validation_data_sheet_fn[:len(validation_data_sheet_fn)-len('.xlsx')] + ' - annotated.xlsx',index=False)