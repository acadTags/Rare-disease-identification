#a simplified version of step3.1 to train and test a phenotype confirmation model for rare disease entity linking.
#this version also allows the comparison among multiple types of BERT models: BERT, BlueBERT, PubMedBERT, SapBERT

from sent_bert_emb_viz_util import load_data, encode_data_tuple, get_model_from_encoding_output, test_model_from_encoding_output
import random
import pickle
from step4_further_results_from_annotations import get_and_display_results
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

if __name__ == '__main__':
    #0. all sorts of settings
    fill_data = True
    verbo_changes = True # output changes or new predictions in the testing data df.

    masked_training = False
    use_doc_struc = True
    print('masked_training:',masked_training)
    print('use_doc_struc:',use_doc_struc)

    #num_sample = 100#len(data_list_tuples) #len(data_list_tuples) for full data #100 for quick checking of the program

    num_of_training_samples = 1000000 # much more than the whole data, i.e. using the whole data # 9000 for a sample of first 9000 data
    print('num_of_training_samples:',num_of_training_samples)
    window_size = 10 # 5 as default or 10, 20 

    num_of_testing_samples = 1073 # 1073 for full testing data; 400 for validation data

    #select the pre-trained model here
    #model_path='C:\\Users\\hdong3\\Downloads\\uncased_L-12_H-768_A-12'; model_name='BERTbase'
    #model_path='C:\\Users\\hdong3\\Downloads\\pubmedBERT_tf_model\\'; model_name = 'pubmedBERT' # coverted to tf from https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/tree/main
    #model_path='C:\\Users\\hdong3\\Downloads\\SapBERT-from-PubMedBERT-fulltext-tf\\'; model_name = 'SapBERT' # coverted to tf from https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext
    model_path='C:\\Users\\hdong3\\Downloads\\NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12\\'; model_name = 'blueBERTnorm'
    #model_path='C:\\Users\\hdong3\\Downloads\\NCBI_BERT_pubmed_mimic_uncased_L-24_H-1024_A-16\\'; model_name = 'blueBERTlarge'
    
    #model_path='/exports/cmvm/eddie/smgphs/groups/hdong3-res/Rare-disease-identification/uncased_L-12_H-768_A-12/'; model_name='BERTbase' 
    #model_path='/exports/cmvm/eddie/smgphs/groups/hdong3-res/Rare-disease-identification/pubmedBERT_tf_model/'; model_name = 'pubmedBERT'
    #model_path='/exports/cmvm/eddie/smgphs/groups/hdong3-res/Rare-disease-identification/SapBERT-from-PubMedBERT-fulltext-tf/'; model_name = 'SapBERT'
    #model_path = '/exports/cmvm/eddie/smgphs/groups/hdong3-res/Rare-disease-identification/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12/'; model_name = 'blueBERTnorm'
    #model_path = '/exports/cmvm/eddie/smgphs/groups/hdong3-res/Rare-disease-identification/NCBI_BERT_pubmed_mimic_uncased_L-24_H-1024_A-16/'; model_name = 'blueBERTlarge'
        
    marking_str_tr = 'training'

    marking_str_te = 'testing_%d' % num_of_testing_samples

    trained_model_name = 'model_%s_ws%s%s%s.pik' % (model_name, str(window_size), '_ds' if use_doc_struc else '','_masked' if masked_training else '')
    print('trained_model_name:',trained_model_name)

    #1. load data, encoding, and train model 
    #load data
    data_list_tuples = load_data('mention_disamb_data.pik')
    random.Random(1234).shuffle(data_list_tuples) #randomly shuffle the list with a random seed        
    #data_list_tuples = data_list_tuples[0:num_sample] # set a small sample for quick testing
    print(len(data_list_tuples))
    training_data_len = int(len(data_list_tuples)* 0.9)

    print('start encoding')

    #encoding
    output_tuple = encode_data_tuple(data_list_tuples, masking=masked_training, with_doc_struc=use_doc_struc, model_path=model_path, marking_str=marking_str_tr, window_size=window_size)
    print(output_tuple)
    #training
    clf_model = get_model_from_encoding_output(output_tuple,num_of_training_samples)

    #export models
    with open(trained_model_name, 'ab') as data_f:
        pickle.dump(clf_model, data_f)
        print('\n' + trained_model_name, 'saved')

    #2. load testing data and predict results: 
    #load data from .xlsx and save the results to a specific column
    # get a list of data tuples from an annotated .xlsx file
    # data format: a 6-element tuple of section_text, document_structure_name, mention_name, UMLS_code, UMLS_desc, label (True or False)
    df = pd.read_excel('for validation - SemEHR ori.xlsx')

    data_list_tuples = []
    for i, row in df.iterrows():
        doc_struc = row['document structure']
        text = row['Text']
        mention = row['mention']
        UMLS_code = row['UMLS with desc'].split()[0]
        UMLS_desc = ' '.join(row['UMLS with desc'].split()[1:])
        label = row['gold text-to-UMLS label']
        label = 0 if label == -1 else label # assume that the inapplicable (-1) entries are all False.
        #print(label)
        data_tuple = (text,doc_struc,mention,UMLS_code,UMLS_desc,label)
        #if i<2:
        #    print(data_tuple)
        data_list_tuples.append(data_tuple)

    # get testing data rep and predict with the model
    output_tuple_test = encode_data_tuple(data_list_tuples, masking=masked_training, with_doc_struc=use_doc_struc, model_path=model_path, marking_str=marking_str_te, window_size=window_size)
    # prediction
    print('single model results')
    print('%smasked training%s:' % ('' if masked_training else 'non-', ' with doc struc' if use_doc_struc else ''))
    y_test, y_pred_test,list_of_err_samples = test_model_from_encoding_output(output_tuple_test, num_of_testing_samples, clf_model) #also returned the list of erroneous samples using the non-masked encoding.    
    get_and_display_results(y_test, y_pred_test)

    if fill_data:
        print('df_length:',len(df))
        print('y_pred_test:',len(y_pred_test))
        #fill the prediction into the .xlsx file
        ind_y_pred_test=0
        result_column_name = 'model %s prediction%s%s%s%s' % (model_name, ' (masked training)' if masked_training else '', ' ds' if use_doc_struc else '', ' cw%d' % window_size if window_size != 5 else '', ' tr%s' % str(num_of_training_samples) if num_of_training_samples<training_data_len else '')
        if not result_column_name in df.columns:
            df[result_column_name] = ""
        for i, row in df.iterrows():
            if i in list_of_err_samples:
                continue
            if verbo_changes and row[result_column_name] != y_pred_test[ind_y_pred_test]:
                print('row %s results changed %s to %s' % (str(i), row[result_column_name], y_pred_test[ind_y_pred_test]))
            df.at[i,result_column_name] = y_pred_test[ind_y_pred_test]
            ind_y_pred_test = ind_y_pred_test + 1
            if ind_y_pred_test == len(y_pred_test):
                break # terminate the df row loop if the last element of the y_pred_test has been reached.
        df.to_excel('for validation - SemEHR ori - predicted%s%s.xlsx' % (' - masked' if masked_training else '', ' - ds' if use_doc_struc else ''),index=False)