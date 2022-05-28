# get results from Google Healthcare API: an experiment with Google healthcare API to validate the results of SemEHR.

from test_google_medcat_api_util import *
from sent_bert_emb_viz_util import get_context_window#, get_char_offset
import pandas as pd
from tqdm import tqdm
import sys
import json            

#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    with open(file_name, 'w', encoding="utf-8-sig") as f_output:
        f_output.write(str + '\n')
        
if __name__ == '__main__':
    # update the access token before running this
    # simply paste the output of the command here: gcloud auth application-default print-access-token
    gcloud_access_token = '[put your token here]'
    
    gAPI_json_path = './google_API_processed_test_jsons'

    load_from_json = True # not using Google API, but loading from the saved json files only
    
    #setting
    window_size = 5 #50 # 5
    tolerance = 15
    
    #load the sheet to validate
    df = pd.read_excel('for validation - SemEHR ori.xlsx')
    #get a smaller set for quick testing
    #df = df[21:40]
    #df = df[673:675]
    #get those previously as empty
    #df = df[~((df['Google Healthcare API UMLS with desc cw5 new'] == '[]') & (df['Google Healthcare API mention info cw5 new'] == '[]'))]
    #df = df[989]
    print(len(df), 'to process')
    #sys.exit(0)
    
    #query API on a context window for each instance and fill the result to the columns in the sheet
    #uncomment if running for the first time
    df['Google Healthcare API UMLS with desc cw%s%s' % (str(window_size),' t' + str(tolerance) if tolerance != 0 else '')] = ''
    df['Google Healthcare API UMLS with desc cw%s%s' % (str(window_size),' t' + str(tolerance) if tolerance != 0 else '')] = df['Google Healthcare API UMLS with desc cw%s%s' % (str(window_size),' t' + str(tolerance) if tolerance != 0 else '')].apply(list) # convert the column format to list
    df['Google Healthcare API mention info cw%s%s' % (str(window_size),' t' + str(tolerance) if tolerance != 0 else '')] = ''
    df['Google Healthcare API mention info cw%s%s' % (str(window_size),' t' + str(tolerance) if tolerance != 0 else '')] = df['Google Healthcare API mention info cw%s%s' % (str(window_size),' t' + str(tolerance) if tolerance != 0 else '')].apply(list) # convert the column format to list
    
    #df['Google Healthcare API confidence'] = ''
    #df['Google Healthcare API confidence'] = df['Google Healthcare API confidence'].apply(list) # convert the column format to list
    for i, row in tqdm(df.iterrows()):
        #for re-runing the err ones.
        #if i != 989:
        #    continue
        #print(type(row['Google Healthcare API UMLS with desc cw%s' % str(window_size)]))
        #continue
        
        text = row['Text']
        mention_cue = row['mention']
        umls_with_desc = row['UMLS with desc']
        umls_semEHR = umls_with_desc.split()[0]
        
        #get context window
        cw, ment_begin_offset, ment_end_offset = get_context_window(mention_cue,text,window_size=window_size)
        #cw, left_window_size , right_window_size = get_context_window(mention_cue,text,window_size=5)
        #get_context_window(mention_cue, text_snippet_full, window_size, masking=False)
        #print(cw,left_window_size , right_window_size)
        print(cw)
        
        #get character offset of the mention
        # ment_begin_offset = 0
        # tokens_cw = cw.split()
        # for ind, token in enumerate(tokens_cw):
            # if ind < left_window_size:
                # ment_begin_offset = ment_begin_offset + len(token) + 1 # also add the space
        # ment_end_offset = ment_begin_offset + len(mention_cue) - 1
        #ment_begin_offset, ment_end_offset = get_char_offset(cw, left_window_size , right_window_size)
        print(mention_cue, ment_begin_offset, ment_end_offset)
        
        doc_id = row['doc row ID'][1:-1] # remove the [] sign
        if not load_from_json:
            #query google API to get the json output
            json_output = get_json_google_healthcare_api(cw,gcloud_access_token)
            json_output_str = json_output.text
            print(json_output_str)
            err_msg = check_if_error(json_output_str)
            #save the output json file
            filename_gAPI_json = '%s/%s-doc-%s-gAPI%s.json' % (gAPI_json_path, str(i), str(doc_id), '-' + err_msg if err_msg != '' else '')
            #output_to_file(filename_gAPI_json,json_output_str)
            with open(filename_gAPI_json, 'w', encoding='utf-8') as f:
                json.dump(json_output.json(), f, ensure_ascii=False, indent=2)
        else:
            filename_gAPI_json = '%s/%s-doc-%s-gAPI.json' % (gAPI_json_path, str(i), str(doc_id))
            try:
                with open(filename_gAPI_json) as json_file:
                    json_file.seek(0)
                    json_output_str = json_file.read()
                    #print(json_output_str)
            except FileNotFoundError:
                print(filename_gAPI_json, 'not found')
                json_output_str = '{}'
                
        #get entities and process entities
        dict_umls_ids_confi_google_API, dict_mention_google_API = get_entities(json_output_str,mention_cue, ment_begin_offset, ment_end_offset, tolerance=tolerance)
        
        list_umls_ids_google_API = list(dict_umls_ids_confi_google_API.keys())
        #list_confi_sc_google_API = [dict_umls_ids_confi_google_API[umls_id] for umls_id in list_umls_ids_google_API]
        list_mention_info_google_API = list(dict_mention_google_API.keys())
        #list_umls_codes_google_API = [umls_id2cui(umls_id) for umls_id in list_umls_ids_google_API]
        list_umls_codes_google_API = [umls_id[5:] for umls_id in list_umls_ids_google_API]
        list_umls_desc_google_API = get_umls_desc(json_output_str,list_umls_ids_google_API)
        #print(umls_with_desc)
        #print(list_umls_codes_google_API)
        #print(list_umls_desc_google_API)
        
        list_umls_with_desc_google_API = [umls_code + ' ' + umls_desc for umls_code,umls_desc in zip(list_umls_codes_google_API,list_umls_desc_google_API)]
        #print(list_umls_with_desc_google_API)
        
        # if the SemEHR UMLS code is in the set of Google API code, then say yes.
        df.at[i,'Google Healthcare API cw%s%s' % (str(window_size),' t' + str(tolerance) if tolerance != 0 else '')] = 1 if umls_semEHR in list_umls_codes_google_API else 0
        df.at[i,'Google Healthcare API UMLS with desc cw%s%s' % (str(window_size),' t' + str(tolerance) if tolerance != 0 else '')] = list_umls_with_desc_google_API
        df.at[i,'Google Healthcare API mention info cw%s%s' % (str(window_size),' t' + str(tolerance) if tolerance != 0 else '')] = list_mention_info_google_API
        #df.at[i,'Google Healthcare API confidence'] = list_confi_sc_google_API[0] if len(list_confi_sc_google_API)>0 else [] #same confidence score for all UMLS entities linked to the same mention, so we just display the first entry in this list.
        
    df.to_excel('for validation - SemEHR ori - google API.xlsx',index=False)