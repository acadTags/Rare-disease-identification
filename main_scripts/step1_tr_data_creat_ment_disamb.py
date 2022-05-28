# create the training data for mention disambiguation from SemEHR

from sent_bert_emb_viz_util import load_df, retrieve_section_and_doc_structure
from rare_disease_id_util import umls2prefLabelwithDict
from collections import defaultdict
import pickle
import sys
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="step2 - create the training data for mention disambiguation from SemEHR")
    parser.add_argument('-d','--dataset', type=str,
                    help="dataset name", default='MIMIC-III')
    parser.add_argument('-dc','--data-category', type=str,
                    help="category of data", default='Discharge summary')
    args = parser.parse_args()
    
    #dataset='Tayside' # 'MIMIC-III' or 'Tayside'
    #data_category = 'Radiology' # 'Radiology' or 'Discharge summary'
    if args.dataset=='Tayside':
        assert args.data_category=='Radiology' # the report type should be radiology for Tayside data
    
    if args.dataset == 'MIMIC-III':
        data_pik_all_doc = 'df_MIMIC-III DS-Rare-Disease-ICD9-new-rowsNone%s.pik' % ('-rad' if args.data_category == 'Radiology' else '')
    else:
        data_pik_all_doc = 'df_%s-rad-Rare-Disease-rowsNone.pik' % args.dataset
    # else:
        # print('dataset unknown:', dataset)
        # sys.exit(0)
    
    print('loading data: %s' % data_pik_all_doc)
    df = load_df(data_pik_all_doc)    
    #df = load_df(filename='df_MIMIC-III DS-Rare-Disease-ICD9-new-rowsNone.pik')
    
    list_section_retrieved_with_umls = retrieve_section_and_doc_structure(df,data_category=args.data_category,dataset=args.dataset)
    #(text_snippet_full,doc_structure,mention,umls_code,umls_desc)
    
    #0. setting
    parameter_tuning = True # whether or not performing parameter tuning
    prevalence_percentage_threshold_range = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1] if parameter_tuning else [0.005]
    #prevalence_percentage_threshold = 0.005 # 0.005 this should be the prevalence of rare disease in the ICU-admitted patients. 
    # Prevalence of rare disease in the US is about 7.5/10000, see [1] J. Textoris and M. Leone, ‘Genetic Aspects of Uncommon Diseases’, in Uncommon Diseases in the ICU, M. Leone, C. Martin, and J.-L. Vincent, Eds. Cham: Springer International Publishing, 2014, pp. 3–11.
    in_text_matching_len_threshold_range = [2,3,4] if parameter_tuning else [3]
    #in_text_matching_len_threshold = 3 # to avoid wrong SemEHR UMLS matching due to meanings in abbreviations and short length
    
    filter_by_preflabel = False
    
    for prevalence_percentage_threshold in prevalence_percentage_threshold_range:
        for in_text_matching_len_threshold in in_text_matching_len_threshold_range:
            print('p %s l %s' % (prevalence_percentage_threshold,in_text_matching_len_threshold))
            #1. filtering out UMLS_code based on prevalency threshold
            dict_umls_code_freq = defaultdict(int)
            dict_umls_code_dict_desc = defaultdict(lambda: defaultdict(int)) # umls_code -> a dict: umls_desc -> freq
            dict_umls_code_mention_label = defaultdict(lambda: defaultdict(int)) # a default dict of default dict
            for section_with_umls in list_section_retrieved_with_umls:
                mention, umls_code, umls_desc = section_with_umls[2:]
                
                dict_umls_code_freq[umls_code] += 1

                dict_umls_desc_freq = dict_umls_code_dict_desc[umls_code]
                dict_umls_desc_freq[umls_desc] += 1
                
                dict_mention_umls_desc = dict_umls_code_mention_label[umls_code]
                dict_mention_umls_desc[mention + ';' + umls_desc] += 1
                
            num_doc = len(list_section_retrieved_with_umls) # the num_doc is the number of sections
            # store the filtered UMLS_code into *dict_umls_selected_prevlance*
            dict_umls_selected_prevlance = {}
            for umls_code, ann_freq in dict_umls_code_freq.items():
                if ann_freq/float(num_doc) <= prevalence_percentage_threshold:
                    dict_umls_selected_prevlance[umls_code] = 1
                else:
                    #print those not selected by the `prevalence' rule
                    umls_desc = list(dict_umls_code_dict_desc[umls_code].keys())[0]
                    print(umls_code, umls_desc)
                    #print(umls_code + ' ' + umls_desc, dict_umls_code_mention_label[umls_code])
            ##print those selected by the `prevalence' rule
            #print(len(dict_umls_selected_prevlance),len(dict_umls_code_freq),float(len(dict_umls_selected_prevlance))/len(dict_umls_code_freq))
            
            print('display the selected umls_codes by prevalence threshold')
            for umls_code in dict_umls_selected_prevlance.keys():
                ann_freq = dict_umls_code_freq[umls_code]
                dict_umls_desc_freq = dict_umls_code_dict_desc[umls_code]
                assert len(dict_umls_desc_freq) == 1
                umls_desc = list(dict_umls_code_dict_desc[umls_code].keys())[0]
                print(umls_code + ' ' + umls_desc, dict_umls_desc_freq, ann_freq)
            
            #export the umls selected in dict_umls_selected_prevlance 
            #to note that this file changes when the parameter changes
            with open('dict_umls_selected_prevlance.pik', 'wb') as data_f:
                pickle.dump(dict_umls_selected_prevlance, data_f)
                print('\n' + 'dict_umls_selected_prevlance.pik', 'saved')
            #sys.exit(0) # stop at here to just get the dict_umls_selected_prevlance.pik
            
            #2. filtering by in text matching length and exact matching to the preflabel
            dict_mention_length_filtered_out = defaultdict(int)
            dict_preflabel_filtered_out = defaultdict(int)
            dict_umls2preflabel = {}
            for section_with_umls in list_section_retrieved_with_umls:
                section_text, doc_structure, mention, umls_code, umls_desc = section_with_umls
                #(i) filtering out matching where mention is equal to or below certain length, default as 3
                #    and store the mentions filtered out by mention length to *dict_mention_length_filtered_out*
                if len(mention) <= in_text_matching_len_threshold:
                    dict_mention_length_filtered_out[mention] += 1
                else:
                    if filter_by_preflabel:
                        # (ii) filtering out by not matching to the preferred label
                        preflabel, dict_umls2preflabel = umls2prefLabelwithDict(umls_code,dict_umls2preflabel)
                        if preflabel.lower() != mention.lower():
                            dict_preflabel_filtered_out[umls_code] += 1
            #display the filtered out umls_codes
            print('filtered out by short mention length <=%s' % str(in_text_matching_len_threshold))
            for mention, freq in dict_mention_length_filtered_out.items():
                print(mention, freq)
            print('filtered out by exact matching to preflabel')
            for umls_code, freq in dict_preflabel_filtered_out.items():
                print(umls_code, freq, dict_umls_code_mention_label[umls_code])
            
            #3. create positive and negative data
            '''positive: (i) AND (ii), (i) len(mention) > in_text_matching_len_threshold; (ii) ann_freq/float(num_doc) <= prevalence_percentage_threshold
               negative: (NOT (i)) AND (NOT (ii))
               
               we then use *dict_umls_selected_prevlance* and *dict_mention_length_filtered_out* to implement the selection of positive and negative datasets
               
               data format: a tuple of section_text, document_structure_name, mention_name, UMLS_code, UMLS_desc, label (True or False)
            '''
            num_pos, num_neg = 0, 0
            data_weakly_labelled_list_of_tuple = []
            print('all unlabelled data:',len(list_section_retrieved_with_umls))
            for section_with_umls in list_section_retrieved_with_umls:
                section_text, doc_structure, mention, umls_code, umls_desc = section_with_umls
                criteria_i_men_len = dict_mention_length_filtered_out.get(mention,None) == None
                criteria_ii_preval_th = dict_umls_selected_prevlance.get(umls_code,None) != None
                if criteria_i_men_len and criteria_ii_preval_th:
                    data_weakly_labelled_list_of_tuple.append(section_with_umls + (True,))
                    num_pos += 1
                if (not criteria_i_men_len) and (not criteria_ii_preval_th):
                    data_weakly_labelled_list_of_tuple.append(section_with_umls + (False,))
                    num_neg += 1
            print('positive data: %s \nnegative data: %s' % (str(num_pos),str(num_neg)))
            
            if num_pos > 0 and num_neg > 0:
                # save the data with pickle when there are data for both positive and negative classes
                data_list_tuples_pik_fn = 'mention_disamb_data%s%s%s%s.pik' % ('' if args.dataset == 'MIMIC-III' else '_%s' % args.dataset,'-rad' if args.data_category == 'Radiology' else '','_p%s' % str(prevalence_percentage_threshold) if prevalence_percentage_threshold != 0.005 else '','_l%s' % str(in_text_matching_len_threshold) if in_text_matching_len_threshold != 3 else '')
                with open(data_list_tuples_pik_fn, 'wb') as data_f: # only add the thresholds to the file name if the thresholds do not equal to the default values
                    pickle.dump(data_weakly_labelled_list_of_tuple, data_f)