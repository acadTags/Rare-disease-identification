from sent_bert_emb_viz_util import load_data, encode_data_tuple_ave_word_emb, get_model_from_encoding_output, test_model_from_encoding_output
import random
import pickle
from step4_further_results_from_annotations import get_and_display_results
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

#0. all sorts of settings
fill_data = True
verbo_changes = False # output changes or new predictions in the testing data df.

masked_training = False
use_doc_struc = True
print('masked_training:',masked_training)
print('use_doc_struc:',use_doc_struc)

#num_sample = 100 #len(data_list_tuples) #500 #len(data_list_tuples) #1000 for quick checking of the program

# data selection approaches: random sampling, balanced random sampling, and diverse sampling
num_of_training_samples = 1000000 # much more than the whole data, i.e. using the whole data # 9000 for a sample of first 9000 data
print('num_of_training_samples:',num_of_training_samples)
window_size = 5

num_of_testing_samples = 1073 # 1073 for full testing data; 400 for validation data

#select your word embeddings here
#word2vec_model_path = 'processed_full.w2v'; model_name = 'w2v_caml_100'
#word2vec_model_path = 'MIMIC_word_embeddings_300d.txt'; model_name = 'w2v_mimic_300' # this also has min_count as 5 that filtered out the vocab appeared less than 5 times.
word2vec_model_path = 'word-emb-mimic3-768.model'; model_name = 'w2v_mimic_768'

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
output_tuple_w2v = encode_data_tuple_ave_word_emb(data_list_tuples, word2vec_model_path=word2vec_model_path, masking=masked_training, with_doc_struc=use_doc_struc, marking_str=marking_str_tr+'_'+model_name if model_name != 'w2v_caml_100' else marking_str_tr, window_size=window_size, store_encoding_pik=True) # the marking_str contains both training/testing mark and the embedding model name mark (if the model_name is not the default w2v_caml_100).
print(output_tuple_w2v)
#training
clf_model_w2v = get_model_from_encoding_output(output_tuple_w2v,num_of_training_samples)

#export models
with open(trained_model_name, 'ab') as data_f:
    pickle.dump(clf_model_w2v, data_f)
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
output_tuple_test_w2v = encode_data_tuple_ave_word_emb(data_list_tuples, word2vec_model_path=word2vec_model_path, masking=masked_training, with_doc_struc=use_doc_struc, marking_str=marking_str_te+'_'+model_name if model_name != 'w2v_caml_100' else marking_str_te, window_size=window_size)

# prediction
print('single model results')
print('%smasked training%s:' % ('' if masked_training else 'non-', ' with doc struc' if use_doc_struc else ''))
y_test, y_pred_test,list_of_err_samples = test_model_from_encoding_output(output_tuple_test_w2v, num_of_testing_samples, clf_model_w2v) #also returned the list of erroneous samples using the non-masked encoding.    
get_and_display_results(y_test, y_pred_test)

if fill_data:
    print('df_length:',len(df))
    print('y_pred_test:',len(y_pred_test))
    #fill the prediction into the .xlsx file
    ind_y_pred_test=0
    result_column_name = 'model %s prediction%s%s%s' % (model_name, ' (masked training)' if masked_training else '', ' ds' if use_doc_struc else '', ' tr%s' % str(num_of_training_samples) if num_of_training_samples<training_data_len else '')
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
    df.to_excel('for validation - SemEHR ori - predicted - w2v%s%s.xlsx' % (' - masked' if masked_training else '', ' - ds' if use_doc_struc else ''),index=False)