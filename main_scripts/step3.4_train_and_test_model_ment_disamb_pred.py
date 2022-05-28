# the program to train and test model and do a rule-based ensembling of the masked and non-masked models, with or without document structure, for rare disease entity linking filtering.
# this program also export the models for later use in the subsequent steps of this project - to be used in step9_processing_all_documents.py.
# this program also supports the training of models from scratch for radiology reports.

# this program also supports the filling of predictions to the evaluation data sheet.

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sent_bert_emb_viz_util import load_data, encode_data_tuple, get_model_from_encoding_output, test_model_from_encoding_output
import random

import pickle

from evaluation_util import get_and_display_results,rule_based_model_ensemble
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="step2 - create the training data for mention disambiguation from SemEHR")
    parser.add_argument('-d','--dataset', type=str,
                    help="dataset name", default='Tayside')
    parser.add_argument('-dc','--data-category', type=str,
                    help="category of data", default='Discharge summary')
    parser.add_argument('-f','--fill-data', help='fill the predictions on the evaluation sheet',action='store_true',default=False)
    parser.add_argument('-p','--prevalence-percentage-threshold', type=float, required=False, dest='prevalence_percentage_threshold', default=0.005,
                        help="corpus-based prevalence threshold")
    parser.add_argument('-l','--mention-character-length-threshold', dest="mention_character_length_threshold", type=int, required=False, default=3,
                        help="mention character length threshold")
    parser.add_argument('-u','--use-default-param', help='use default paramters for radiology',action='store_true',default=False)  
    parser.add_argument('-b','--balanced-random-sampling', help='use balanced random sampling',action='store_true',default=False)
    parser.add_argument('-ts','--num-test-samples', type=int, required=False, dest='num_test_samples',default=400,help="the first ts number of testing samples on the evaluation sheet")
    parser.add_argument('-t','--test', help='whether test on evaluation sheet - the gold annotation is needed in advance',action='store_true',default=False)  
    args = parser.parse_args()
    
    dataset=args.dataset # 'MIMIC-III' or 'Tayside' or 'GS-Tay'
    data_category = args.data_category # 'Radiology' or 'Discharge summary'
    if dataset=='Tayside' or dataset[:3] == 'GS-':
        assert data_category=='Radiology' # the report type should be radiology for Tayside data
            
    fill_data = args.fill_data
    
    use_default_param_for_rad = args.use_default_param # whether or not to use default paramters (p0.005, l3) from discharge summaires for radiology reports
    
    get_and_report_key_model = False if args.data_category == 'Discharge summary' else True # if set as true, only report the non-masked ds model for discharge summary and non-masked model for radiology
    
    #test_with_non_rule_annotated_only = False
    #num_of_testing_items = 50 # this applies only when test_with_non_annotated_only is False

    #num_sample = 100 #len(data_list_tuples) #500 #len(data_list_tuples) #1000 for quick checking of the program

    # data selection approaches: random sampling, balanced random sampling, and diverse sampling
    
    num_of_training_samples_non_masked_ds = 9000 # 9000 (best tuned for use_doc_struc as True)    
    num_of_training_samples_non_masked = 10000000 # 1000000, much more than the whole data, i.e. using the whole data (for use_doc_struc as False)
    num_of_training_samples_masked = 500 # much more than the whole data, i.e. using the whole data # 500
    #balanced_random_sampling=False
    diverse_sampling=False
    num_of_data_per_mention=25 # for diverse sampling only
    
    masking_rate = 1 #0.15 # to implement
    window_size = 5
    C = 1 # l2 regularisation parameter

    num_of_testing_samples = args.num_test_samples
    
    model_path = './bert-models/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12/'; model_name = 'blueBERTnorm'   
    #model_path='./bert-models/pubmedBERT_tf_model_new/microsoft/'; model_name = 'pubmedBERT'
    #model_path = './bert-models/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12/'; model_name = 'blueBERTnorm'
    #model_path = './bert-models/NCBI_BERT_pubmed_mimic_uncased_L-24_H-1024_A-16/'; model_name = 'blueBERTlarge'
    
    #setting parameters      
    if use_default_param_for_rad or data_category == 'Discharge summary':
        prevalence_percentage_threshold = args.prevalence_percentage_threshold # "prevalence" threshold, p
        in_text_matching_len_threshold = args.mention_character_length_threshold # mention length threshold, l
        balanced_random_sampling=args.balanced_random_sampling
    elif data_category == 'Radiology':
        if dataset == 'MIMIC-III':
            #for best F1
            prevalence_percentage_threshold = 0.0005 # "prevalence" threshold, p
            in_text_matching_len_threshold = 4 # mention length threshold, l
            #for best recall
            #prevalence_percentage_threshold = 0.01 # "prevalence" threshold, p
            #in_text_matching_len_threshold = 4 # mention length threshold, l
            balanced_random_sampling=True
        elif dataset == 'Tayside':
            #for best F1
            #prevalence_percentage_threshold = 0.0005 # "prevalence" threshold, p
            #in_text_matching_len_threshold = 4 # mention length threshold, l
            #for best recall
            prevalence_percentage_threshold = 0.01 # "prevalence" threshold, p
            in_text_matching_len_threshold = 4 # mention length threshold, l
            balanced_random_sampling=False # non-balanced sampling worked better for Tayside data
        
    #the default model (p as 0.005 and l as 3)
    #choose a model with other parameters in weak supervision
    #data_list_tuples_pik_fn = 'mention_disamb_data_p0.0001.pik'
    #marking_str_tr = 'training_p0.0001'
    #marking_str_te = 'testing_200'    
    #marking_str_te = 'testing_non_rule_anno'
    #marking_str_te = 'testing_198_MIMIC-III_rad' if dataset == 'MIMIC-III' else 'testing_280_Tayside_rad'
    if data_category == 'Discharge summary':
        data_list_tuples_pik_fn = 'mention_disamb_data%s%s.pik' % ('_p%s' % str(prevalence_percentage_threshold) if prevalence_percentage_threshold != 0.005 else '','_l%s' % str(in_text_matching_len_threshold) if in_text_matching_len_threshold != 3 else '')
        marking_str_tr = 'training'
        marking_str_te = 'testing_1073'    
        evaluation_data_sheet_fn = 'for validation - SemEHR ori.xlsx'
    elif data_category == 'Radiology':
        data_list_tuples_pik_fn = 'mention_disamb_data%s%s%s%s.pik' % ('' if dataset == 'MIMIC-III' else '_%s' % dataset,'-rad','_p%s' % str(prevalence_percentage_threshold) if prevalence_percentage_threshold != 0.005 else '','_l%s' % str(in_text_matching_len_threshold) if in_text_matching_len_threshold != 3 else '')
        marking_str_tr = 'training_rad' if dataset == 'MIMIC-III' else 'training_%s_rad' % dataset    
        if dataset == 'MIMIC-III':
            marking_str_te = 'testing_198_MIMIC-III_rad'
        elif dataset == 'Tayside':
            marking_str_te = 'testing_279_Tayside_rad'
        else:
            marking_str_te = 'testing_%s_rad' % dataset
        evaluation_data_sheet_fn = 'for validation - 1000 docs - ori - MIMIC-III-rad.xlsx' if dataset == 'MIMIC-III' else 'for validation - 5000 docs - ori - %s - rad.xlsx' % dataset.lower()
    else:
        print('data category unknown:', data_category)
        sys.exit(0)
    
    #add paramter marks to the marking_str_tr
    marking_str_tr = marking_str_tr + '%s%s' % ('_p%s' % str(prevalence_percentage_threshold) if prevalence_percentage_threshold != 0.005 else '','_l%s' % str(in_text_matching_len_threshold) if in_text_matching_len_threshold != 3 else '')
    print('marking_str_tr:',marking_str_tr)
    
    #trained_models_name = './models/model_%s_ws%s%s%s%s.pik' % (model_name, str(window_size), '_ds' if use_doc_struc else '', '_divs%s' % str(num_of_data_per_mention) if diverse_sampling else '','_nm%sm%s' % (str(num_of_training_samples_non_masked),str(num_of_training_samples_masked)))
    trained_models_name = './models/model%s%s_%s_ws%s%s%s%s.pik' % ('' if dataset == 'MIMIC-III' else '_%s' % dataset,'_rad' if data_category == 'Radiology' else '',model_name, str(window_size), '_divs%s' % str(num_of_data_per_mention) if diverse_sampling else '','_p%s' % str(prevalence_percentage_threshold) if prevalence_percentage_threshold != 0.005 else '','_l%s' % str(in_text_matching_len_threshold) if in_text_matching_len_threshold != 3 else '')
    #print(trained_models_name)

    #1. load data, encoding, and train model 
    #load data
    #data_list_tuples = load_data('mention_disamb_data.pik')
    data_list_tuples = load_data(data_list_tuples_pik_fn)
    random.Random(1234).shuffle(data_list_tuples) #randomly shuffle the list with a random seed        
    #data_list_tuples = data_list_tuples[0:num_sample] # set a small sample for quick testing
    print(len(data_list_tuples))

    print('start encoding')

    #encoding
    output_tuple_masked = encode_data_tuple(data_list_tuples, masking=True, with_doc_struc=False, model_path=model_path, marking_str=marking_str_tr, window_size=window_size, masking_rate=masking_rate, diverse_sampling=diverse_sampling, num_of_data_per_mention=num_of_data_per_mention) if not get_and_report_key_model else None
    output_tuple_non_masked_ds = encode_data_tuple(data_list_tuples, masking=False, with_doc_struc=True, model_path=model_path, marking_str=marking_str_tr, window_size=window_size, masking_rate=masking_rate, diverse_sampling=diverse_sampling, num_of_data_per_mention=num_of_data_per_mention) if (not get_and_report_key_model) or (data_category == 'Discharge summary') else None
    output_tuple_non_masked = encode_data_tuple(data_list_tuples, masking=False, with_doc_struc=False, model_path=model_path, marking_str=marking_str_tr, window_size=window_size, masking_rate=masking_rate, diverse_sampling=diverse_sampling, num_of_data_per_mention=num_of_data_per_mention) if (not get_and_report_key_model) or (data_category == 'Radiology') else None

    #training
    clf_model_masked = get_model_from_encoding_output(output_tuple_masked,num_of_training_samples_masked,balanced_sampling=balanced_random_sampling) if output_tuple_masked != None else None
    clf_model_non_masked_ds = get_model_from_encoding_output(output_tuple_non_masked_ds,num_of_training_samples_non_masked_ds,balanced_sampling=balanced_random_sampling) if (not get_and_report_key_model) or (data_category == 'Discharge summary') else None
    clf_model_non_masked = get_model_from_encoding_output(output_tuple_non_masked,num_of_training_samples_non_masked,balanced_sampling=balanced_random_sampling) if (not get_and_report_key_model) or (data_category == 'Radiology') else None
    
    #export models - this is for the subsequent steps in the pipeline
    with open(trained_models_name, 'wb') as data_f:
        pickle.dump((clf_model_non_masked_ds,clf_model_masked,clf_model_non_masked), data_f)
        print('\n' + trained_models_name, 'saved')
    
    if args.test:
        #2. load testing data and predict results: 
        #load data from .xlsx and save the results to a specific column
        # get a list of data tuples from an annotated .xlsx file
        # data format: a 6-element tuple of section_text, document_structure_name, mention_name, UMLS_code, UMLS_desc, label (True or False)
        df = pd.read_excel(evaluation_data_sheet_fn)
        # change nan values into empty strings in the two rule-based label columns
        #df[['neg label: only when both rule 0','pos label: both rules applied']] = df[['neg label: only when both rule 0','pos label: both rules applied']].fillna('') 
        # if the data is not labelled, i.e. nan, label it as -1 (not positive or negative)
        #df[['manual label from ann1']] = df[['manual label from ann1']].fillna(-1) 
        
        if not 'gold text-to-UMLS label' in df.columns:
            print('error: gold annotation is not available, test ending')
            sys.exit()
        data_list_tuples = []
        for i, row in df.iterrows():
            #filter out the manually added rows that were created during the annotation
            if 'manually added data' in row:
                if not pd.isna(row['manually added data']):
                    print('row %s is a manually added datum based on annotation： ignored in data_list_tuples for encoding' % i)
                    continue
            doc_struc = row['document structure']
            text = row['Text']
            mention = row['mention']
            UMLS_code = row['UMLS with desc'].split()[0]
            UMLS_desc = ' '.join(row['UMLS with desc'].split()[1:])
            #label = row['manual label from ann1']
            label = row['gold text-to-UMLS label']
            label = 0 if label == -1 else label # assume that the inapplicable (-1) entries are all False.
            #print(label)
            data_tuple = (text,doc_struc,mention,UMLS_code,UMLS_desc,label)
            #if i<2:
            #    print(data_tuple)
            data_list_tuples.append(data_tuple)
            
        # get testing data rep and predict with the model
        # encoding
        output_tuple_test_masked = encode_data_tuple(data_list_tuples, masking=True, with_doc_struc=False, model_path=model_path, marking_str=marking_str_te, window_size=window_size, masking_rate=masking_rate,port_number_str='5555') if not get_and_report_key_model else None
        output_tuple_test_non_masked_ds = encode_data_tuple(data_list_tuples, masking=False, with_doc_struc=True, model_path=model_path, marking_str=marking_str_te, window_size=window_size, masking_rate=masking_rate,port_number_str='5555') if (not get_and_report_key_model) or (data_category == 'Discharge summary') else None
        output_tuple_test_non_masked = encode_data_tuple(data_list_tuples, masking=False, with_doc_struc=False, model_path=model_path, marking_str=marking_str_te, window_size=window_size, masking_rate=masking_rate,port_number_str='5555') if (not get_and_report_key_model) or (data_category == 'Radiology') else None

        # prediction
        if get_and_report_key_model:
            if data_category == 'Discharge summary':
                print('non-masked training with ds:')
                y_test_labelled_non_masked_ds, y_pred_test_labelled_non_masked_ds,list_of_err_samples_non_masked_ds = test_model_from_encoding_output(output_tuple_test_non_masked_ds, num_of_testing_samples, clf_model_non_masked_ds)
                get_and_display_results(y_test_labelled_non_masked_ds, y_pred_test_labelled_non_masked_ds)
                #also returned the list of erroneous samples using the non-masked encoding.
            else:
                print('non-masked training:')
                y_test_labelled_non_masked, y_pred_test_labelled_non_masked,list_of_err_samples_non_masked = test_model_from_encoding_output(output_tuple_test_non_masked, num_of_testing_samples, clf_model_non_masked)
                get_and_display_results(y_test_labelled_non_masked, y_pred_test_labelled_non_masked)
        else:
            print('single model results')
            print('masked training:')
            y_test_labelled_masked, y_pred_test_labelled_masked,list_of_err_samples_masked = test_model_from_encoding_output(output_tuple_test_masked, num_of_testing_samples, clf_model_masked)
            get_and_display_results(y_test_labelled_masked, y_pred_test_labelled_masked)

            print('non-masked training with ds:')
            y_test_labelled_non_masked_ds, y_pred_test_labelled_non_masked_ds,list_of_err_samples_non_masked_ds = test_model_from_encoding_output(output_tuple_test_non_masked_ds, num_of_testing_samples, clf_model_non_masked_ds)
            get_and_display_results(y_test_labelled_non_masked_ds, y_pred_test_labelled_non_masked_ds)
            #also returned the list of erroneous samples using the non-masked encoding.
            
            print('non-masked training:')
            y_test_labelled_non_masked, y_pred_test_labelled_non_masked,list_of_err_samples_non_masked = test_model_from_encoding_output(output_tuple_test_non_masked, num_of_testing_samples, clf_model_non_masked)
            get_and_display_results(y_test_labelled_non_masked, y_pred_test_labelled_non_masked)
            
        
            #rule-based ensembling
            y_pred_ment_len = df[['rule (mention length >3)']].to_numpy()
            y_pred_prevalence = df[['rule (prevalance th <= 0.005)']].to_numpy()

            y_pred_ment_len_labelled = np.array([y_pred_ment_len[ind] for ind in range(len(y_pred_ment_len)) if ind<num_of_testing_samples and ind not in list_of_err_samples_non_masked_ds])
            print('y_pred_ment_len_labelled:',len(y_pred_ment_len_labelled))
            # here we ignored the possibility that if some testing data not labelled to 0 or 1.

            y_pred_prevalence_labelled = np.array([y_pred_prevalence[ind] for ind in range(len(y_pred_prevalence)) if ind<num_of_testing_samples and ind not in list_of_err_samples_non_masked_ds])
            print('y_pred_prevalence_labelled:',len(y_pred_prevalence_labelled))

            print('rule-based model ensemble best scenario results:')
            # #y_pred_test_m_labelled_ensemb = np.logical_or(y_pred_test_m_labelled,y_pred_test_m_ds_large_labelled).astype(int)
            y_pred_rule_based_model_ensemble = rule_based_model_ensemble(y_pred_ment_len_labelled, y_pred_prevalence_labelled, y_pred_test_labelled_non_masked_ds, y_pred_test_labelled_masked)
            get_and_display_results(y_test_labelled_non_masked_ds, y_pred_rule_based_model_ensemble)

        if fill_data:
            #parameters for filling data
            masked_training = False
            use_doc_struc = False if data_category == 'Radiology' else True
            
            y_pred_test = y_pred_test_labelled_non_masked_ds if use_doc_struc else y_pred_test_labelled_non_masked # no ds for radiology reports
            num_of_training_samples = num_of_training_samples_non_masked_ds if use_doc_struc else num_of_training_samples_non_masked
            
            if masked_training:
                list_of_err_samples = list_of_err_samples_masked 
            else:
                list_of_err_samples = list_of_err_samples_non_masked_ds if use_doc_struc else list_of_err_samples_non_masked
                   
            print('df_length:',len(df))
            print('y_pred_test:',np.array(y_pred_test).shape)
            #fill the prediction into the .xlsx file
            ind_y_pred_test=0
            result_column_name = 'model %s prediction%s%s%s%s%s' % (model_name, ' (masked training)' if masked_training else '', ' ds' if use_doc_struc else '', ' tr%s' % str(num_of_training_samples) if num_of_training_samples<10000 else '', ' p%s' % str(prevalence_percentage_threshold) if prevalence_percentage_threshold != 0.005 else '', ' l%s' % str(in_text_matching_len_threshold) if in_text_matching_len_threshold != 3 else '') # display the number of training data if there is a selection by numbers (e.g. less than 10k)
            print('updating data (if necessary) for %s' % result_column_name)        
            if not result_column_name in df.columns:
                df[result_column_name] = ""
            ind_non_manual = 0
            for i, row in df.iterrows():
                 #filter out the manually added rows that were created during the annotation (for Tayside data)
                if 'manually added data' in row:
                    if not pd.isna(row['manually added data']):
                        df.at[i,result_column_name] = 0 # set the pred as 0 if this is a manually added row  
                        print('row %s is a manually added datum based on annotation' % i)
                        continue
                if ind_non_manual in list_of_err_samples:
                    df.at[i,result_column_name] = 0 # set the pred as 0 if there is an error (we use this rule just for Tayside data)
                    print('row %s results set to 0 due to encoding error' % str(i))
                    ind_non_manual = ind_non_manual + 1
                    continue
                if row[result_column_name] != y_pred_test[ind_y_pred_test]:
                    print('row %s results changed %s to %s' % (str(i), row[result_column_name], y_pred_test[ind_y_pred_test]))
                df.at[i,result_column_name] = y_pred_test[ind_y_pred_test]
                ind_non_manual = ind_non_manual + 1
                ind_y_pred_test = ind_y_pred_test + 1
            df.to_excel(evaluation_data_sheet_fn[:len(evaluation_data_sheet_fn)-len('.xlsx')] + ' - predicted%s%s.xlsx' % (' - masked' if masked_training else '', ' - ds' if use_doc_struc else ''),index=False)