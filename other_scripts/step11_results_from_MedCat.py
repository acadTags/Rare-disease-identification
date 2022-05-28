# get results from MedCAT to validate the results of SemEHR.

from test_google_medcat_api_util import get_entities_from_MedCAT_outputs,get_umls_desc,output_to_file,check_if_error,get_umls_desc_MedCAT
from sent_bert_emb_viz_util import get_context_window#, get_char_offset
import pandas as pd
from tqdm import tqdm
import sys
import json            
import os

from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.cat import CAT
from medcat.meta_cat import MetaCAT
from medcat.preprocessing.tokenizers import TokenizerWrapperBPE
from tokenizers import ByteLevelBPETokenizer

def fix_float32_type_err(metacat_dict_with_float32):
    '''fix the TypeError: Object of type float32 is not JSON serializable
    due to the meta_anns confidence score is float32, not supported by JSON'''
    #based on https://stackoverflow.com/questions/55704719/python-replace-values-in-nested-dictionary
    for entity in metacat_dict_with_float32['entities'].values():
        entity['meta_anns']['Status']['confidence'] = float(entity['meta_anns']['Status']['confidence'])

if __name__ == '__main__':
    load_from_json = True # set this as False if running the program for the first time. If set as True, then not running MedCAT to annotate, but loading from the saved json files only
    use_meta_ann = False # whether using meta_annotations 
    #list_window_size = [5,10] # window size options as a list to test
    list_window_size = [5]
    #list_model_size = ['small','medium'] #'medium', 'small'
    list_model_size = ['medium']
    #list_model_size = ['large']
    
    #tolerance in matching API detected offsets to gold mention offsets.
    tolerance = 0 # default as 0
    
    # update the access token before running this
    MedCAT_json_path = './MedCAT_processed_test_jsons'
    
    # create the output folder if not existed
    if not os.path.exists(MedCAT_json_path):
        os.makedirs(MedCAT_json_path)
    
    #load the sheet to validate
    df = pd.read_excel('for validation - SemEHR ori.xlsx')
    print(len(df), 'to process')
    
    for model_size in list_model_size:
        # # Create cat - each cdb comes with a config that was used
        # #to train it. You can change that config in any way you want, before or after creating cat.
        if not load_from_json:    
            # Load the vocab model you downloaded
            vocab = Vocab.load(r'./medcat models/vocab.dat')
            # Load the cdb model you downloaded
            if model_size == 'small':
                cdb = CDB.load(r'./medcat models/cdb-umls-sm-v1.dat') # Small, 500k concepts (diseases, medications, symptoms, findings, substances and a couple of other), RAM requirement 3GB (Performance for existing concepts on par with Medium)
            elif model_size == 'medium':
                cdb = CDB.load(r'./medcat models/cdb_mimic_md_21-April-2021.dat') # Medium, ~4M concepts, RAM requirement ~12GB (1-2% worse performance than Large)
            elif model_size == 'large':    
                cdb = CDB.load(r'./medcat models/cdb-umls-v1.dat') # Large, ~4M concepts, RAM requirement ~20GB
            
            # IMPORTANT: Set TUI filters
            # |T047|Disease or Syndrome
            # |T048|Mental or Behavioral Dysfunction
            #tui_filter = ['T047', 'T048']
            tui_filter = ['T047']
            cui_filters = set()
            for tui in tui_filter:
              cui_filters.update(cdb.addl_info['type_id2cuis'][tui])
            cdb.config.linking['filters']['cuis'] = cui_filters
            print(f"The size of the cdb is now: {len(cui_filters)}")
            
            if use_meta_ann:
                mc_status = MetaCAT.load("mc_status")            
                cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab, meta_cats=[mc_status])
            else:
                cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab)
        for window_size in list_window_size:
            #query API on a context window for each instance and fill the result to the columns in the sheet
            #uncomment if running for the first time
            df['MedCAT UMLS with desc cw%s %s%s%s' % (str(window_size), model_size, ' t' + str(tolerance) if tolerance != 0 else '', ' metaAnn' if use_meta_ann else '')] = ''
            df['MedCAT UMLS with desc cw%s %s%s%s' % (str(window_size), model_size, ' t' + str(tolerance) if tolerance != 0 else '', ' metaAnn' if use_meta_ann else '')] = df['MedCAT UMLS with desc cw%s %s%s%s' % (str(window_size), model_size, ' t' + str(tolerance) if tolerance != 0 else '', ' metaAnn' if use_meta_ann else '')].apply(list) # convert the column format to list
            df['MedCAT mention info cw%s %s%s%s' % (str(window_size), model_size, ' t' + str(tolerance) if tolerance != 0 else '', ' metaAnn' if use_meta_ann else '')] = ''
            df['MedCAT mention info cw%s %s%s%s' % (str(window_size), model_size, ' t' + str(tolerance) if tolerance != 0 else '', ' metaAnn' if use_meta_ann else '')] = df['MedCAT mention info cw%s %s%s%s' % (str(window_size), model_size, ' t' + str(tolerance) if tolerance != 0 else '', ' metaAnn' if use_meta_ann else '')].apply(list) # convert the column format to list
            
            for i, row in tqdm(df.iterrows()):
                text = row['Text']
                mention_cue = row['mention']
                umls_with_desc = row['UMLS with desc']
                umls_semEHR = umls_with_desc.split()[0]
                
                #get context window
                cw, ment_begin_offset, ment_end_offset = get_context_window(mention_cue,text,window_size=window_size)
                print(cw)
                print(mention_cue, ment_begin_offset, ment_end_offset)
                
                doc_id = row['doc row ID'][1:-1] # remove the [] sign
                if not load_from_json:
                    #query MedCAT to get the json output
                    doc_catted = cat.get_entities(cw)
                    print(type(doc_catted),doc_catted)
                    if use_meta_ann: # fix metacat json float32 type err
                        fix_float32_type_err(doc_catted)
                    json_output_str = json.dumps(doc_catted, indent = 2)
                    #print(json_output_str)
                    #err_msg = check_if_error(json_output_str)
                    err_msg = ''
                    #save the output json file
                    filename_MedCAT_json = '%s/%s-doc-%s-cw%s-%s-MedCAT%s%s.json' % (MedCAT_json_path, str(i), str(doc_id), str(window_size),model_size, '-metaAnn' if use_meta_ann else '','-' + err_msg if err_msg != '' else '')
                    with open(filename_MedCAT_json, 'w', encoding='utf-8') as f:
                        json.dump(doc_catted, f, ensure_ascii=False, indent=2)
                else:
                    filename_MedCAT_json = '%s/%s-doc-%s-cw%s-%s-MedCAT%s.json' % (MedCAT_json_path, str(i), str(doc_id),str(window_size),model_size, '-metaAnn' if use_meta_ann else '')
                    try:
                        with open(filename_MedCAT_json) as json_file:
                            json_file.seek(0)
                            json_output_str = json_file.read()
                            #print(json_output_str)
                    except FileNotFoundError:
                        print(filename_MedCAT_json, 'not found')
                        json_output_str = '{}'
                        
                #get entities and process entities
                dict_umls_ids_confi_MedCAT, dict_mention_MedCAT = get_entities_from_MedCAT_outputs(json_output_str,mention_cue, ment_begin_offset, ment_end_offset,tolerance=tolerance,acc_threshold=0,use_meta_ann=use_meta_ann) # set acc_threshold to 0, no filtering by acc here.
                
                print(dict_umls_ids_confi_MedCAT, dict_mention_MedCAT)
                list_umls_ids_MedCAT = list(dict_umls_ids_confi_MedCAT.keys())
                list_mention_info_MedCAT = list(dict_mention_MedCAT.keys())
                list_umls_desc_MedCAT = get_umls_desc_MedCAT(json_output_str,list_umls_ids_MedCAT)
                
                list_umls_with_desc_MedCAT = [umls_code + ' ' + umls_desc for umls_code,umls_desc in zip(list_umls_ids_MedCAT,list_umls_desc_MedCAT)]
                
                # if the SemEHR UMLS code is in the set of Google API code, then say yes.
                df.at[i,'MedCAT cw%s %s%s%s' % (str(window_size), model_size, ' t' + str(tolerance) if tolerance != 0 else '', ' metaAnn' if use_meta_ann else '')] = dict_umls_ids_confi_MedCAT[umls_semEHR] if umls_semEHR in list_umls_ids_MedCAT else 0 # here store the confidence score
                df.at[i,'MedCAT UMLS with desc cw%s %s%s%s' % (str(window_size), model_size, ' t' + str(tolerance) if tolerance != 0 else '', ' metaAnn' if use_meta_ann else '')] = list_umls_with_desc_MedCAT
                df.at[i,'MedCAT mention info cw%s %s%s%s' % (str(window_size), model_size, ' t' + str(tolerance) if tolerance != 0 else '', ' metaAnn' if use_meta_ann else '')] = list_mention_info_MedCAT
                            
        df.to_excel('for validation - SemEHR ori - MedCAT - %s%s.xlsx' % (' '.join(list_model_size),' - metaAnn' if use_meta_ann else ''),index=False)