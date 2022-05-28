# predict with all documents (with either a weak or a strong supervised model) to get the rare diseases

from sent_bert_emb_viz_util import load_df,load_data, encode_data_tuple, retrieve_section_and_doc_structure, test_model_from_encoding_output
from rare_disease_id_util import umls2ordoListFromCSV, isNotGroupOfDisorders, umls2ICD9FromCSV, ICD92ORDOListFromCSV, union, intersection,get_ORDO_pref_label_from_CSV#, hasICD9linkage2ORDO
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sys
import argparse

# avoid ... in showing long sequence
pd.set_option("display.max_colwidth", 10000)

if __name__ == '__main__':
    #0.settings
    parser = argparse.ArgumentParser(description="step3 - apply pre-trained phenotype confirmation models to new radiology reports")
    parser.add_argument('-d','--dataset', type=str,
                    help="dataset name", default='Tayside')
    parser.add_argument('-dc','--data-category', type=str,
                    help="category of data", default='Discharge summary')
    parser.add_argument('-st','--supervision-type', type=str,
                    help="type of supervision: weak or strong", default='weak')
    parser.add_argument('-u','--use-default-param', 
                    help='use default paramters for radiology',action='store_true',default=False)
    parser.add_argument('-p','--prevalence-percentage-threshold', type=float, required=False, dest='prevalence_percentage_threshold', default=0.005,
                        help="corpus-based prevalence threshold")
    parser.add_argument('-l','--mention-character-length-threshold', dest="mention_character_length_threshold", type=int, required=False, default=3,
                        help="mention character length threshold")
    parser.add_argument('-om','--ontology-matching-source', type=str, 
                    help='ontology matching source for ORDO to ICD-9 matching: (i) NZ (default, ordo-*icd10-icd9* with MoH NZ); (ii) bp (ORDO-*UMLS-icd9* with bioportal ICD-9-CM); (iii) from both NZ and bp sources, \'both\'.',default='both')                
    parser.add_argument('-en','--exact-or-narrower-only', 
                    help='whether using exact or narrower only matching from ORDO to ICD-10 (if setting to True, this will result in a set of rare disease ICD-9 codes with higher precision but lower recall. This will affect sources of \'NZ\' and \'both\'.)',action='store_true',default=False)                    
    parser.add_argument('-trans','--direct-transfer', 
                    help='whether to direct transfer a weakly supervised phenotype confirmation model.)',action='store_true',default=False)                    
    parser.add_argument('-wsm','--weak-supervision-model-path', type=str,
                    help="only used when direct transfer (i.e. --trans), weak supervision model path, default as trained from MIMIC-III discharge summaries", default='./models/model_blueBERTnorm_ws5.pik') 
    parser.add_argument('-ssm','--strong-supervision-model-path', type=str,
                    help="strong supervision model path, default as trained from manual annotation (validation data) from MIMIC-III discharge summaries", default='./models/model_blueBERTnorm_ws5_sup.pik')
    #parser.add_argument('-m','--masked-training', help='mention masking in encoding',action='store_true',default=False)
    #parser.add_argument('-ds','--use-document-structure', help='use document structure in encoding',action='store_true',default=False)
    
    args = parser.parse_args()
    
    dataset=args.dataset # 'MIMIC-III' or 'Tayside'
    data_category = args.data_category # 'Radiology' or 'Discharge summary'
    if dataset=='Tayside':
        assert data_category=='Radiology' # the report type should be radiology for Tayside data
    
    pred_model_type = args.supervision_type # 'weak' or 'strong'
    use_default_param_for_rad = args.use_default_param
    if pred_model_type == 'weak':
        #setting parameters for weak supervision model
        if use_default_param_for_rad or data_category == 'Discharge summary':
            prevalence_percentage_threshold = args.prevalence_percentage_threshold # "prevalence" threshold, p
            in_text_matching_len_threshold = args.mention_character_length_threshold # mention length threshold, l
            #balanced_random_sampling=False
        elif data_category == 'Radiology':
            if dataset == 'MIMIC-III':
                #for best recall
                prevalence_percentage_threshold = 0.01 # "prevalence" threshold, p
                in_text_matching_len_threshold = 4 # mention length threshold, l
                #balanced_random_sampling=True
            elif dataset == 'Tayside':
                #for best F1
                #prevalence_percentage_threshold = 0.0005 # "prevalence" threshold, p
                #in_text_matching_len_threshold = 4 # mention length threshold, l
                #for best recall
                prevalence_percentage_threshold = 0.01 # "prevalence" threshold, p
                in_text_matching_len_threshold = 4 # mention length threshold, l
                #balanced_random_sampling=False # non-balanced sampling worked better for Tayside data
            
    bert_model_path='./bert-models/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12/'; model_name = 'blueBERTnorm' # for encoding
    
    icd9_matching_onto_source = args.ontology_matching_source # 'NZ', 'bp', or 'both' as icd9 matching onto path: (i) NZ (default, ordo-*icd10-icd9* with MoH NZ); (ii) bp (ORDO-*UMLS-icd9* with bioportal ICD-9-CM); (iii) from both NZ and bp sources, 'both'.
    exact_or_narrower_only = args.exact_or_narrower_only # whether using exact or narrower only matching from ORDO to ICD-10 (if setting to True, this will result in a set of rare disease ICD-9 codes with higher precision but lower recall. This will affect sources of 'NZ' and 'both'.)
    
    save_full_text=False if dataset=='MIMIC-III' else True #not recommended to save the MIMIC-III full texts as a column in the output file (as they are too long), but for Tayside this is okay
    
    # for selecting only first k rows
    n_rows_selected = None #None #1000, 5000, None
    
    #1. load trained weak/strong supervision models
    if pred_model_type == 'weak':
        #select the weakly supervised model trained from different datasets
        if args.direct_transfer:
            trained_model_name = args.weak_supervision_model_path
        else:
            trained_model_name = './models/model%s%s_blueBERTnorm_ws5%s%s.pik' % ('' if dataset=='MIMIC-III' else '_%s' % dataset, '_rad' if data_category == 'Radiology' else '','_p%s' % str(prevalence_percentage_threshold) if prevalence_percentage_threshold != 0.005 else '','_l%s' % str(in_text_matching_len_threshold) if in_text_matching_len_threshold != 3 else '')
        print('model to use:',trained_model_name)    
        if os.path.exists(trained_model_name):
            with open(trained_model_name, 'rb') as data_f:
                clf_non_masked_ds, _, clf_non_masked = pickle.load(data_f) # not loading the masked model
    elif pred_model_type == 'strong':
        #strongly supervised model only trained from MIMIC-III discharge summaries
        #trained_model_name = 'model_blueBERTnorm_ws5_ds_sup.pik' # strongly supervised non-masked & document structure model
        trained_model_name = args.strong_supervision_model_path # 2-model tuple: strongly supervised non-masked & document structure model + strongly supervised non-masked model
        print('model to use:',trained_model_name)
        if os.path.exists(trained_model_name):
            with open(trained_model_name, 'rb') as data_f:
                clf_non_masked_ds, clf_non_masked = pickle.load(data_f)
    else:
        print('pred_model_type wrong, neither weak or strong, value:%s' % pred_model_type)
        sys.exit(0)
    
    clf_model = clf_non_masked if data_category == 'Radiology' else clf_non_masked_ds # clf_non_masked for radiology datasets as no document structured information was parsed; clf_non_masked_ds for discharge summaries as there were document structure information.
    
    use_doc_struc = False if data_category == 'Radiology' else True # not using document structure information when the report type is radiology
    
    #2. load data
    
    #data_pik_all_doc = 'df_MIMIC-III DS-Rare-Disease-ICD9-new-rowsNone.pik' # the output from step0
    if dataset == 'MIMIC-III':
        data_pik_all_doc = 'df_MIMIC-III DS-Rare-Disease-ICD9-new-rowsNone%s.pik' % ('-rad' if data_category == 'Radiology' else '')
    else:
        data_pik_all_doc = 'df_%s-rad-Rare-Disease-rowsNone.pik' % dataset
    #else:
    #    print('dataset unknown:', dataset)
    #    sys.exit(0)
        
    print('loading data: %s' % data_pik_all_doc)
    df = load_df(data_pik_all_doc)
    df = df[:n_rows_selected] if n_rows_selected != None else df #only process k docs, otherwise process all docs (by setting n_rows_selected as None)
    #get the mention-umls pairs, each mention is in its section (or document structure), row_id is included as the first element of the output tuples in the list
    list_section_retrieved_with_umls = retrieve_section_and_doc_structure(df,dataset=dataset,data_category=data_category,with_row_id=True)
    
    #3. prediction: text to UMLS
    print('prediction: text to UMLS')
    #encoding
    data_list_tuple_selected = [tuple_ele[1:] + (True,) for tuple_ele in list_section_retrieved_with_umls] # not counting the first ele of the tuple, which is the row id. and add the fake label (as True) as required in function encode_data_tuple()
    print(len(data_list_tuple_selected))
    print('data_list_tuple_selected[0]:',data_list_tuple_selected[0])
    output_tuple_test = encode_data_tuple(data_list_tuple_selected, masking=False, with_doc_struc=use_doc_struc, model_path=bert_model_path, marking_str='all_data%s%s' % ('' if dataset == 'MIMIC-III' else '_%s' % dataset, '_rad' if data_category == 'Radiology' else ''), window_size=5, masking_rate=1,port_number_str='5556') #specify a different port number as previously (it was '5555')
       
    _,pred_t_U,list_ind_err_test = test_model_from_encoding_output(output_tuple_test,len(data_list_tuple_selected),clf_model)    
    pred_t_U = np.asarray(pred_t_U)
    # insert label for the err index data as 0.
    for ind_err in list_ind_err_test:
        pred_t_U = np.insert(pred_t_U,ind_err,0)        
    #print(pred_t_U)
    
    #4. prediction: UMLS to ORDO
    print('prediction: UMLS to ORDO')
    pred_U_O = np.zeros(len(list_section_retrieved_with_umls))
    # load saved dictionary files (.pik), if not exist initialise empty dicts.
    if os.path.exists('dict_UMLS_ORDOlist.pik'):
        with open('dict_UMLS_ORDOlist.pik', 'rb') as data_f:
            dict_UMLS_ORDOlist=pickle.load(data_f)
    else:
        dict_UMLS_ORDOlist = {} # dictionary from a UMLS to a tuple of the list of ORDO IDs and the corresponding list of ORDO pref labels
    updated_dict_UMLS_ORDOlist = False # whether the dict updated
    map=None # initialise map for code matching    
    if os.path.exists('dict_ORDO_not_GoD.pik'):
        with open('dict_ORDO_not_GoD.pik', 'rb') as data_f:
            dict_ORDO_not_GoD=pickle.load(data_f)
    else:
        dict_ORDO_not_GoD = {} # dictionary ORDO not GoD.
    updated_dict_ORDO_not_GoD = False # whether the dict updated
    list_list_ORDO_IDs = [] # a list of matched ORDO_ID_lists for each mention-level data
    list_list_ORDO_pref_labels = [] # the corresponding pref labels for the matched ORDO_ID_lists
    for ind, section_retrived_with_umls in tqdm(enumerate(list_section_retrieved_with_umls)):
        _, _, _, _, umls_code, _ = section_retrived_with_umls
        #add rare disease phenotype information if available
        #if pred[ind] == 1:
        
        # get the corresponding ORDO_ID (as a list, but usually just one and can be more than one) from the umls_code
        if dict_UMLS_ORDOlist.get(umls_code,None) == None:
            list_ORDO_ID_tmp, list_ORDO_pref_label_tmp, map = umls2ordoListFromCSV(umls_code,map=map)            
            dict_UMLS_ORDOlist[umls_code] = list_ORDO_ID_tmp, list_ORDO_pref_label_tmp
            updated_dict_UMLS_ORDOlist = True
        list_ORDO_IDs, list_ORDO_pref_labels = dict_UMLS_ORDOlist[umls_code]
        list_ORDO_IDs = [ORDO_ID[26:] for ORDO_ID in list_ORDO_IDs] #need to ensure only one matched or to deal with all the matched ORDOs - done
        list_list_ORDO_IDs.append(list_ORDO_IDs)
        list_list_ORDO_pref_labels.append(list_ORDO_pref_labels)
        #if len(list_ORDO_IDs) > 1:
        #    print('more than one ORDO matched:', len(list_ORDO_IDs), 'for', umls_code, 'at', ind)
        for ORDO_ID in list_ORDO_IDs:
            #ORDO_ID = ORDO_ID[26:]
            # filter the non-exact-matching ORDO_ID out by rule 
            # add the filtering information (not_GoD) to the dictionary
            if dict_ORDO_not_GoD.get(ORDO_ID,None) == None:
                dict_ORDO_not_GoD[ORDO_ID] = isNotGroupOfDisorders(ORDO_ID)
                updated_dict_ORDO_not_GoD = True
            if dict_ORDO_not_GoD[ORDO_ID]:
                pred_U_O[ind] = 1
                break # break when at least one ORDO in the list satisfy the rule (not a Group of Disorder type in the ORDO ontology)
                #to do - here above to list all results instead of just for one ORDO
                
    #update the list_section_retrieved_with_umls with matched ORDO IDs
    list_section_retrieved_with_umls_ordos = [section_retrived_with_umls + (list_ORDO_IDs,list_ORDO_pref_labels) for section_retrived_with_umls,list_ORDO_IDs,list_ORDO_pref_labels in zip(list_section_retrieved_with_umls,list_list_ORDO_IDs,list_list_ORDO_pref_labels)]
    
    #save the dict files with pickle
    if updated_dict_UMLS_ORDOlist:
        with open('dict_UMLS_ORDOlist.pik', 'wb') as data_f:
            pickle.dump(dict_UMLS_ORDOlist, data_f)
    if updated_dict_ORDO_not_GoD:
        with open('dict_ORDO_not_GoD.pik', 'wb') as data_f:
            pickle.dump(dict_ORDO_not_GoD, data_f)
        
    #5. prediction: text to ORDO
    print('prediction: text to ORDO')
    pred_t_O = np.multiply(pred_t_U, pred_U_O)
    #print(pred_t_U, pred_U_O, pred_t_O)
    
    #6. save the mention-level and admission-level results 
    print('save the mention-level results ')
    #for index, row in tqdm(df.iterrows()):
    #save and export mention-level results
    df_mention_level = pd.DataFrame(list_section_retrieved_with_umls_ordos, columns=['doc row ID','Text','document structure','mention','UMLS code', 'UMLS desc', 'ORDO ID list','ORDO pref label list']) # from a list of tuples to a dataframe
    df_mention_level['pred text-to-UMLS'] = pred_t_U.tolist()
    df_mention_level['pred UMLS-to-ORDO'] = pred_U_O.tolist()
    df_mention_level['pred text-to-ORDO'] = pred_t_O.tolist()
    
    df_mention_level.to_excel('mention-level pred results all docs%s%s%s.xlsx' % ('' if dataset == 'MIMIC-III' else '_%s' % dataset, '-rad' if data_category == 'Radiology' else '', ' sup' if pred_model_type=='strong' else ''),index=False)
    
    #save admission-level results (also linking to ICD9)
    print('save the admission-level results ')
    #create dict of row_id -> (UMLS, ORDO, ICD9) from df_mention_level
    dict_row_id_onto = {}
    # for umls to ICD9: load saved dictionary files (.pik), if not exist initialise empty dicts.
    fn_dict_UMLS_ICD9 = 'dict_UMLS_ICD9list_%s%s.pik' % (icd9_matching_onto_source, '_E_N' if exact_or_narrower_only else '')
    if os.path.exists(fn_dict_UMLS_ICD9):
        with open(fn_dict_UMLS_ICD9, 'rb') as data_f:
            dict_UMLS_ICD9list=pickle.load(data_f)
    else:
        dict_UMLS_ICD9list = {} # dictionary from a UMLS to list of ICD9 codes
    updated_dict_UMLS_ICD9list = False # whether the dict updated
    map=None # initialising map for code matching    
    for i, row in tqdm(df_mention_level.iterrows()):
        row_id = row['doc row ID']
        umls_code,umls_desc,list_ORDO_IDs,list_ORDO_pref_labels = row['UMLS code'], row['UMLS desc'], row['ORDO ID list'],row['ORDO pref label list']
        # add the UMLS, ORDO, ICD9 info to the row id if the t-O prediction is 1 (True).
        if row['pred text-to-ORDO'] == 1:
            # get ICD9 info (here we used UMLS to match through our previous umls-ordo-icd10-icd9 mapping)
            if dict_UMLS_ICD9list.get(umls_code,None) == None:
                 list_icd9_tmp, list_icd9_long_tit_tmp, map = umls2ICD9FromCSV(umls_code, map=map, onto_source=icd9_matching_onto_source,exact_or_narrower_only=exact_or_narrower_only)
                 dict_UMLS_ICD9list[umls_code] = list_icd9_tmp, list_icd9_long_tit_tmp
                 updated_dict_UMLS_ICD9list = True
            list_icd9, list_icd9_long_tit = dict_UMLS_ICD9list[umls_code]
            if dict_row_id_onto.get(row_id,None) == None:
                dict_row_id_onto[row_id] = ([umls_code],[umls_desc],list_ORDO_IDs,list_ORDO_pref_labels, list_icd9, list_icd9_long_tit) # make umls_code and umls_desc as lists
            else:
                list_umls_code_prev,list_umls_desc_prev,list_ORDO_IDs_prev,list_ORDO_pref_labels_prev, list_icd9_prev, list_icd9_long_tit_prev = dict_row_id_onto[row_id] # unpack the tuple of lists
                #update the lists of UMLS, ORDO, and ICD9 info with new True mention-level pred
                list_umls_code = list_umls_code_prev if umls_code in list_umls_code_prev else list_umls_code_prev + [umls_code]
                list_umls_desc = list_umls_desc_prev if umls_desc in list_umls_desc_prev else list_umls_desc_prev + [umls_desc]
                list_ORDO_IDs = list_ORDO_IDs_prev + [ORDO_ID for ORDO_ID in list_ORDO_IDs if ORDO_ID not in list_ORDO_IDs_prev]
                list_ORDO_pref_labels = list_ORDO_pref_labels_prev + [ORDO_pref_label for ORDO_pref_label in list_ORDO_pref_labels if ORDO_pref_label not in list_ORDO_pref_labels_prev]
                list_icd9 = list_icd9_prev + [icd9_code for icd9_code in list_icd9 if icd9_code not in list_icd9_prev]
                list_icd9_long_tit = list_icd9_long_tit_prev + [icd9_long_tit for icd9_long_tit in list_icd9_long_tit if icd9_long_tit not in list_icd9_long_tit_prev]
                dict_row_id_onto[row_id] = (list_umls_code,list_umls_desc,list_ORDO_IDs,list_ORDO_pref_labels,list_icd9,list_icd9_long_tit)
    #save the dict of UMLS -> ICD9list
    if updated_dict_UMLS_ICD9list:
        with open(fn_dict_UMLS_ICD9, 'wb') as data_f:
            pickle.dump(dict_UMLS_ICD9list, data_f)
    
    #add the filtered UMLS, ORDO, and ICD9 (as umls_RD_filtered, umls_RD_desc_filtered, ORDO_ID_filtered, ORDO_pref_label_filtered, icd9_RD_filtered, icd9_RD_long_tit_filtered) to df
    df['umls_RD_filtered'] = ""
    df['umls_RD_filtered'] = df['umls_RD_filtered'].apply(list)
    df['umls_RD_desc_filtered'] = ""
    df['umls_RD_desc_filtered'] = df['umls_RD_desc_filtered'].apply(list)
    df['ORDO_ID_filtered'] = ""
    df['ORDO_ID_filtered'] = df['ORDO_ID_filtered'].apply(list)
    df['ORDO_pref_label_filtered'] = ""
    df['ORDO_pref_label_filtered'] = df['ORDO_pref_label_filtered'].apply(list)
    #also add linking to icd9
    #the icd9 codes linked to those identified and filtered by the NLP tool (SemEHR+rule+weak supervision)
    df['icd9_RD_filtered'] = ""
    df['icd9_RD_filtered'] = df['icd9_RD_filtered'].apply(list)
    df['icd9_RD_long_tit_filtered'] = ""
    df['icd9_RD_long_tit_filtered'] = df['icd9_RD_long_tit_filtered'].apply(list)
    for i, row in df.iterrows():
        row_id = row['ROW_ID'] if dataset == 'MIMIC-III' else row['id']
        if dict_row_id_onto.get(row_id,None) != None:
            #unpack the lists of onto info from the dictionary
            list_umls_code,list_umls_desc,list_ORDO_IDs,list_ORDO_pref_labels, list_icd9, list_icd9_long_tit = dict_row_id_onto[row_id]
            df.at[i,'umls_RD_filtered'] = list_umls_code
            df.at[i,'umls_RD_desc_filtered'] = list_umls_desc
            df.at[i,'ORDO_ID_filtered'] = list_ORDO_IDs
            df.at[i,'ORDO_pref_label_filtered'] = list_ORDO_pref_labels
            df.at[i,'icd9_RD_filtered'] = list_icd9
            df.at[i,'icd9_RD_long_tit_filtered'] = list_icd9_long_tit
        
    #7. compare manual icd9 with the ontology-based and *weakly filtered* rare disease icd9 - only for MIMIC-III dataset
    if dataset == 'MIMIC-III':
        print('adding code comparison columns')
        map_d_icd_diagnosis = pd.read_csv('data/D_ICD_DIAGNOSES.csv')

        df['icd9_RD_manual'] = "" # manual icd9 codes linked to any ORDO IDs
        df['icd9_RD_manual'] = df['icd9_RD_manual'].apply(list)
        df['icd9_common_filtered'] = ""
        df['ORDO_ID_icd9_common_filtered'] = "" # the ORDO_IDs that contributes to the common part of icd9 (as ICD9 were matched by UMLS->ORDO->ICD10->ICD9): this is the intersection of the ORDO_ID_filtered and the ORDO_IDs associated to the manual icd9 codes (or, but not just, to the icd9_common_filtered)
        #df['ORDO_pref_label_icd9_common_filtered'] = ""
        df['ORDO_ID_icd9_manual'] = "" # ORDO from the manual ICD9 codes
        df['ORDO_ID_new_in_icd9'] = "" # ORDO from the manual ICD9 codes but not from NLP
        df['ORDO_ID_pref_label_new_in_icd9'] = "" # ORDO pref labels from the manual ICD9 codes but not from NLP
        df['ORDO_ID_pref_label_new_in_icd9'] = df['ORDO_ID_pref_label_new_in_icd9'].apply(list)
        df['ORDO_ID_new_in_NLP'] = "" # ORDO from NLP but not from the manual ICD9 codes        
        df['icd9_new_RD_filtered'] = ""
        df['icd9_common_short_tit_filtered'] = ""
        df['icd9_common_long_tit_filtered'] = ""
        df['icd9_new_RD_short_tit_filtered'] = ""
        df['icd9_new_RD_long_tit_filtered'] = ""
        # for ICD9 to ORDO: load saved dictionary files (.pik), if not exist initialise empty dicts.
        fn_dict_ICD9_ORDO = 'dict_ICD9_ORDOlist_%s%s.pik' % (icd9_matching_onto_source,'_E_N' if exact_or_narrower_only else '')
        if os.path.exists(fn_dict_ICD9_ORDO):
            with open(fn_dict_ICD9_ORDO, 'rb') as data_f:
                dict_ICD9_ORDOlist=pickle.load(data_f)
        else:
            dict_ICD9_ORDOlist = {} # dictionary from an ICD9 to a tuple of (list of ORDO concepts, list of pref_labels)
        updated_dict_ICD9_ORDOlist = False
        map=None #initialise map for coding matching    
        for i, row in tqdm(df.iterrows()):
            list_mimic_icd9 = row['ICD9_CODE']
            #print(list_mimic_icd9, type(list_mimic_icd9)) # type as list
            #df.at[i, 'icd9_RD_manual'] = [code for code in list_mimic_icd9 if hasICD9linkage2ORDO(code)] #loop over the manual codes and check whether they have linkage to any ORDO IDs # this is too slow
            list_rare_disease_icd9 = row['icd9_RD_filtered']
            df.at[i,'icd9_common_filtered'] = [code for code in list_rare_disease_icd9 if code in list_mimic_icd9]
            #get the ORDO_IDs that contributed to the common part of ICD9
            list_ORDO_ID_icd9_common_filtered = []
            ORDO_ID_icd9_manual = []
            #loop over all the ICD9s (not just the common ones) and match each back to a list of ORDO and union the lists.
            for icd9_code in list_mimic_icd9: #df.at[i,'icd9_common_filtered']:
                if dict_ICD9_ORDOlist.get(icd9_code,None) == None:
                     list_ORDO_ID_icd9_tmp, list_ORDO_pref_label_tmp, map = ICD92ORDOListFromCSV(icd9_code,map=map,onto_source=icd9_matching_onto_source,exact_or_narrower_only=exact_or_narrower_only)
                     dict_ICD9_ORDOlist[icd9_code] = (list_ORDO_ID_icd9_tmp, list_ORDO_pref_label_tmp)
                     updated_dict_ICD9_ORDOlist = True
                list_ORDO_ID_icd9_tmp, _ = dict_ICD9_ORDOlist[icd9_code]
                #add the icd9 code to icd9_RD_manual if it has a linkage to any ORDO concept, i.e. at least one matched ORDO ID
                if len(list_ORDO_ID_icd9_tmp) > 0:
                    df.at[i,'icd9_RD_manual'].append(icd9_code)                
                #union the list of matched ORDO IDs to those matched with other icd9 codes from the same discharge summary
                ORDO_ID_icd9_manual = union(ORDO_ID_icd9_manual,list_ORDO_ID_icd9_tmp)
                
            #get the intersection of this set of ORDO_IDs to all the ORDO_IDs matched (all filtered).
            list_ORDO_ID_filtered = row['ORDO_ID_filtered']
            list_ORDO_ID_icd9_common_filtered = intersection(ORDO_ID_icd9_manual,list_ORDO_ID_filtered)
            df.at[i,'ORDO_ID_icd9_manual'] = ORDO_ID_icd9_manual
            df.at[i,'ORDO_ID_icd9_common_filtered'] = list_ORDO_ID_icd9_common_filtered
            df.at[i,'ORDO_ID_new_in_icd9'] = [ORDO_ID for ORDO_ID in ORDO_ID_icd9_manual if not ORDO_ID in list_ORDO_ID_filtered]
            df.at[i,'ORDO_ID_new_in_NLP'] = [ORDO_ID for ORDO_ID in list_ORDO_ID_filtered if not ORDO_ID in ORDO_ID_icd9_manual]
            for ORDO_ID in df.at[i,'ORDO_ID_new_in_icd9']:
                ORDO_pref_label, map = get_ORDO_pref_label_from_CSV(ORDO_ID,map=map)
                df.at[i,'ORDO_ID_pref_label_new_in_icd9'].append(ORDO_pref_label)
            #df.at[i,'ORDO_ID_pref_label_new_in_icd9'] = [get_ORDO_pref_label_from_CSV(ORDO_ID,map=map)[0] for ORDO_ID in ORDO_ID_icd9_manual if not ORDO_ID in list_ORDO_ID_filtered]
            df.at[i,'icd9_new_RD_filtered'] = [code for code in list_rare_disease_icd9 if not code in list_mimic_icd9]
            # map to short titles and long titles
            mapped_df_list_common = [(code,map_d_icd_diagnosis[map_d_icd_diagnosis['ICD9_CODE']==code]) for code in df.at[i,'icd9_common_filtered']]
            mapped_df_list_new_RD = [(code,map_d_icd_diagnosis[map_d_icd_diagnosis['ICD9_CODE']==code]) for code in df.at[i,'icd9_new_RD_filtered']]
            #print(mapped_df_list)
            #if mapped_df_list != []:
            df.at[i,'icd9_common_short_tit_filtered'] = [code + ':' + mapped_df['SHORT_TITLE'].to_string(index=False).strip() if not 'Series([], )' in mapped_df['SHORT_TITLE'].to_string(index=False) else code + ':not found in D_ICD_DIAGNOSES' for code, mapped_df in mapped_df_list_common]
            df.at[i,'icd9_common_long_tit_filtered'] = [code + ':' + mapped_df['LONG_TITLE'].to_string(index=False).strip() if not 'Series([], )' in mapped_df['LONG_TITLE'].to_string(index=False) else code + ':not found in D_ICD_DIAGNOSES' for code, mapped_df in mapped_df_list_common]
            df.at[i,'icd9_new_RD_short_tit_filtered'] = [code + ':' + mapped_df['SHORT_TITLE'].to_string(index=False).strip() if not 'Series([], )' in mapped_df['SHORT_TITLE'].to_string(index=False) else code + ':not found in D_ICD_DIAGNOSES' for code, mapped_df in mapped_df_list_new_RD]
            df.at[i,'icd9_new_RD_long_tit_filtered'] = [code + ':' + mapped_df['LONG_TITLE'].to_string(index=False).strip() if not 'Series([], )' in mapped_df['LONG_TITLE'].to_string(index=False) else code + ':not found in D_ICD_DIAGNOSES' for code, mapped_df in mapped_df_list_new_RD]
        
        #save the dict of ICD9 -> ORDOlist
        if updated_dict_ICD9_ORDOlist:
            with open(fn_dict_ICD9_ORDO, 'wb') as data_f:
                pickle.dump(dict_ICD9_ORDOlist, data_f)
            
    #export df
    if not save_full_text: 
        # drop the full_text column if not saving it
        df = df.drop(columns=['TEXT' if dataset == 'MIMIC-III' else 'report'])
    #save the df to pickle
    with open('df_%s-Rare-Disease-ICD9-new-rows%s%s-filtered%s-%s%s.pik' % ('MIMIC-III DS' if dataset == 'MIMIC-III' else dataset, n_rows_selected, '-rad' if data_category == 'Radiology' else '', '-sup' if pred_model_type=='strong' else '',icd9_matching_onto_source,'-E-N' if exact_or_narrower_only else ''), 'wb') as data_f:
        pickle.dump(df, data_f)        
    #save the df to csv
    #df.to_csv('%s-Rare-Disease-ICD9-new%s-filtered%s-%s%s.csv' % ('MIMIC-III DS' if dataset == 'MIMIC-III' else 'Tayside', '-rad' if data_category == 'Radiology' else '', '-sup' if pred_model_type=='strong' else '', icd9_matching_onto_source,'-E-N' if exact_or_narrower_only else ''),index=False)
    #save the df to excel    
    df.to_excel('%s-Rare-Disease-ICD9-new%s-filtered%s-%s%s.xlsx' % ('MIMIC-III DS' if dataset == 'MIMIC-III' else dataset, '-rad' if data_category == 'Radiology' else '', '-sup' if pred_model_type=='strong' else '', icd9_matching_onto_source,'-E-N' if exact_or_narrower_only else ''),index=False,engine='xlsxwriter' if dataset == 'Tayside' else 'openpyxl') # use xlsxwriter engine for Tayside data to avoid openpyxl.utils.exceptions.IllegalCharacterError.