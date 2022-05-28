# load the pik file and generate the (sampled) data for annotation

from sent_bert_emb_viz_util import load_df
import pandas as pd
import random
import sys
from tqdm import tqdm

if __name__ == '__main__':
    # number of random samples for validation
    data_category = 'Radiology' # 'Radiology' or 'Discharge summary'
    
    num_samples_valid = 500 if data_category == 'Radiology' else 500 #500 for MIMIC-III discharge, 1000 for MIMIC-III rad
    
    data_pik_all_doc = 'df_MIMIC-III DS-Rare-Disease-ICD9-new-rowsNone%s.pik' % ('-rad' if data_category == 'Radiology' else '')
    print('loading data: %s' % data_pik_all_doc)
    df = load_df(data_pik_all_doc)
    num_doc = len(df.index)
    print('num_doc:',num_doc)
    
    # map rare disease UMLS to ORDO and icd9_RD
    map = pd.read_excel('ORDO2UMLS_ICD10_ICD9+titles_final.xlsx')

    # for human validation
    # generate num_samples_valid random examples for validation - the sampling is at the document level; the output is at the section-mention level, but all section-mention in a document is placed consecutively in the output.
    dict_num_rd={} # save the row index to dict
    random.seed(1234) # fix the random seed for reproducibility
    for x in range(num_samples_valid):
        pick = random.randint(0,num_doc-1)
        #while dict_num_rd.get(pick,None) != None: # ensure always distinct to have exact the num_samples_valid documents sampled
        #    pick = random.randint(0,num_doc-1)
        dict_num_rd[pick] = 1

    print(len(dict_num_rd),dict_num_rd)
    #df_for_validation: umls; text; in_text; UMLS with desc; ORDO with desc
    #create a new df from data https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-then-filling-it
    data2valid = []
    for i,row in tqdm(df.iterrows()):
        if dict_num_rd.get(i,None) != None: # here we select the sampled ones
        #if True: # here we simply process all document to create the annotation sheet
            list_umls_texts = row['umls_RD;doc_structure;text_snippet_full;in_text;label']
            row_id = row['ROW_ID']
            if data_category != 'Discharge summary':
                text_full = row['TEXT']
                list_offsets = row['mention_offsets']
                assert len(list_offsets) == len(list_umls_texts)
            for ind, umls_texts in enumerate(list_umls_texts): # here it ignored documents which do not have any mentions matched to an ORDO-filtered UMLS concept.
                match_eles = umls_texts.split(';')
                
                umls_with_desc = match_eles[0] + ' ' + match_eles[-1]
                mention = match_eles[-2]
                doc_structure = match_eles[1]
                if data_category == 'Discharge summary':
                    text_snippet_full = ';'.join(match_eles[2:-2]) 
                else:
                    #mark the mention in the full text based on the offsets
                    pos_start, pos_end = list_offsets[ind].split(' ')
                    pos_start, pos_end = int(pos_start), int(pos_end)
                    text_snippet_full = text_full[:pos_start] + '*****' + text_full[pos_start:pos_end] + '*****' + text_full[pos_end:] #text_full[] # full text if the data is not disch sum
                
                umls_RD_tmp = 'UMLS:' + match_eles[0] + '\''
                matched_df=map[map['UMLS IDs'].str.contains(umls_RD_tmp)] # will this match to multiple rows? - yes it can
                ordo_ID_tmp = matched_df['ORDO ID'].to_string(index=False)
                ordo_pref_label_tmp = matched_df['Preferred Label'].to_string(index=False)
                ordo_with_desc = ordo_ID_tmp[26:] + ' ' + ordo_pref_label_tmp #get rid of the common path part
                
                data2valid.append([row_id,doc_structure,text_snippet_full,mention,umls_with_desc,ordo_with_desc])
            
    df_for_validation = pd.DataFrame(data2valid,columns=['doc row ID','document structure','Text','mention','UMLS with desc', 'ORDO with desc'])

    df_for_validation.to_excel('for validation - %d docs - ori%s.xlsx' % (num_samples_valid,' - rad' if data_category == 'Radiology' else ''),index=False)