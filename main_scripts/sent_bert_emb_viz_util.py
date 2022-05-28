from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
from bert_serving.server.helper import get_shutdown_parser

import pandas as pd
import os
from tqdm import tqdm
import re
import pickle
import sys
import numpy as np

from nltk.tokenize import RegexpTokenizer # tokeniser for w2v average embedding

#from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer

from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import random

# Yield successive n-sized 
# chunks from l. 
def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
        
#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    with open(file_name, 'w', encoding="utf-8-sig") as f_output:
        f_output.write(str + '\n')
        
def load_df(filename='df_MIMIC-III DS-Rare-Disease-ICD9-new.pik'):
    # load df from the .pik file, instead of .xlsx, as .pik perserves the list type in the columns.
    if os.path.exists(filename):
        with open(filename, 'rb') as data_f:
            df=pickle.load(data_f)
    else:
        print('df not exist!')
        sys.exit(0)    
    return df

def load_data(filename='mention_disamb_data.pik'):
    if os.path.exists(filename):
        with open(filename, 'rb') as data_f:
            data=pickle.load(data_f)
    else:
        print('data file not exist!')
        sys.exit(0)    
    return data

def load_data_from_excel(filename='for validation - SemEHR ori.xlsx'):
    df = pd.read_excel(filename,engine='openpyxl')
    return df
    
def save_pik(data, filename):
    # save the file to .pik
    pass 

def sample_document_with_mention(df,mention_cue=None,potential_rare_disease_doc_from_SemEHR=False,num_doc_to_sample=5):
    #sample a list of documents with a mention, 
    #where the mention is included in the document, (input mention_cue)
    #and has a rare disease UMLS identified from SemEHR (potential_rare_disease_doc_from_SemEHR as True).
    #when potential_rare_disease_doc_from_SemEHR set as True, the mention_cue should be in the SemEHR parsed texts.
    
    #output the list of indexes
    mention_cue = ' ' + mention_cue + ' ' # limit this to a exact word search (and word should be within two spaces)
    
    list_index = []
    df = df.sample(frac=1) # shuffle the dataframe rows
    for index, row in df.iterrows():
        if potential_rare_disease_doc_from_SemEHR:
            list_umls_texts = row['umls_RD;doc_structure;text_snippet_full;in_text;label']
            if len(list_umls_texts) == 0:
                continue
            for ind, umls_texts in enumerate(list_umls_texts):
                match_eles = umls_texts.split(';')
                text_snippet_full = ';'.join(match_eles[2:-2])
                if mention_cue in text_snippet_full:
                    #found one, then return the index#
                    list_index.append(index)
                    break
        if mention_cue in row['TEXT']:
            #found one, then return the index#
            list_index.append(index)
            #return index
        if len(list_index) == num_doc_to_sample:
            return list_index
    #not enough sampled, but return as many as it can.
    return list_index
    
def retrieve_section_and_doc_structure(df,dataset='MIMIC-III',data_category='Discharge summary',mention_cue=None,get_a_random=False,with_row_id=False):
    '''
    helper class for data set creation.
    input: dataset, 'MIMIC-III' or others (i.e. 'Tayside')
           data_category, 'Discharge summary' or 'Radiology'
           mention_cue if None, then select all annotations regardless of the mention; otherwise, filter by mention.
           get a random, if True, then select a random document and output the result list; if False, retrieve results from all documents.
           with row id, if True, also return doc row id as the first element of the tuple; if False, not to return the doc row id.
    output a list of tuples: (full section text,
                                document structure name,
                                mention detected,
                                umls code,
                                umls description)
    umls code only contains those linked to ORDO
    also output the index (as the first element of the output tuple) if get_a_random is set as True'''                       
    list_section_retrieved_with_umls = []
    if get_a_random:
        df = df.sample(frac=1) # shuffle the dataframe rows
    for index, row in tqdm(df.iterrows()):
        list_umls_texts = row['umls_RD;doc_structure;text_snippet_full;in_text;label']
        row_id = row['ROW_ID'] if dataset == 'MIMIC-III' else row['id']
        if data_category != 'Discharge summary':
            text_full = row['TEXT'] if dataset == 'MIMIC-III' else row['report']
            list_offsets = row['mention_offsets']
            assert len(list_offsets) == len(list_umls_texts)
        if len(list_umls_texts) == 0:
            continue        
        for ind, umls_texts in enumerate(list_umls_texts):
            #print(index,umls_texts)
            match_eles = umls_texts.split(';')
            umls_desc = match_eles[-1]
            mention = match_eles[-2]
            doc_structure = match_eles[1]
            umls_code = match_eles[0] 
            
            if mention_cue != None:
                if mention.lower() == mention_cue.lower():
                    # there is a match                
                    # get text_snippet_full
                    if data_category == 'Discharge summary':
                        text_snippet_full = ';'.join(match_eles[2:-2]) 
                    else:
                        #mark the mention in the full text based on the offsets
                        pos_start, pos_end = list_offsets[ind].split(' ')
                        pos_start, pos_end = int(pos_start), int(pos_end)
                        text_snippet_full = text_full[:pos_start] + '*****' + text_full[pos_start:pos_end] + '*****' + text_full[pos_end:] #text_full[] # full text if the data is not disch sum                        
                    if with_row_id:
                        list_section_retrieved_with_umls.append((row_id,text_snippet_full,doc_structure,mention,umls_code,umls_desc))
                    else:
                        list_section_retrieved_with_umls.append((text_snippet_full,doc_structure,mention,umls_code,umls_desc))
                    if get_a_random: # it is time to break if just get one random document results
                        return index, list_section_retrieved_with_umls
            else:
                #select all annotations regardless of mention
                # get text_snippet_full
                if data_category == 'Discharge summary':
                    text_snippet_full = ';'.join(match_eles[2:-2]) 
                else:
                    #mark the mention in the full text based on the offsets
                    pos_start, pos_end = list_offsets[ind].split(' ')
                    pos_start, pos_end = int(pos_start), int(pos_end)
                    text_snippet_full = text_full[:pos_start] + '*****' + text_full[pos_start:pos_end] + '*****' + text_full[pos_end:] #text_full[] # full text if the data is not disch sum
                if with_row_id:
                    list_section_retrieved_with_umls.append((row_id,text_snippet_full,doc_structure,mention,umls_code,umls_desc))
                else:
                    list_section_retrieved_with_umls.append((text_snippet_full,doc_structure,mention,umls_code,umls_desc))
                if get_a_random: # it is time to break if just get one random document results
                    return index, list_section_retrieved_with_umls
    if get_a_random:
        # no document is found with this mention, thus return index as -1 and list_section_retrieved_with_umls as []
        return -1, list_section_retrieved_with_umls
    return list_section_retrieved_with_umls
    
def retrieve_sent_and_doc_structure_from_mention(df,mention_cue,window_size=3):
    '''output a list of tuples: (context window,
                                start of position of the mention in context, 
                                end of position of the mention in context,
                                document structure name,
                                umls description)'''
    list_sent_retrieved_with_pos_tuple = []
    for index, row in df.iterrows():
        list_umls_texts = row['umls_RD;doc_structure;text_snippet_full;in_text;label']
        if len(list_umls_texts) == 0:
            continue        
        for ind, umls_texts in enumerate(list_umls_texts):
            #print(index,umls_texts)
            match_eles = umls_texts.split(';')
            umls_desc = match_eles[-1]
            mention = match_eles[-2]
            doc_structure = match_eles[1]
            
            if mention.lower() == mention_cue.lower():
                # there is a match                
                # get text_snippet_full and the context window
                text_snippet_full = ';'.join(match_eles[2:-2])
                text_snippet_context_window, start_pos_in_ctnx,end_pos_in_ctnx = get_context_window(mention_cue.lower(),text_snippet_full,window_size=window_size)
                if text_snippet_context_window != '': # if things go well (it should be), i.e. there is exactly one mention in the sentence.
                    #list_sent_retrieved.append(doc_structure + ': ' + text_snippet_context_window)        
                    list_sent_retrieved_with_pos_tuple.append((text_snippet_context_window,start_pos_in_ctnx,end_pos_in_ctnx,doc_structure,umls_desc))                            
    return list_sent_retrieved_with_pos_tuple

def get_context_window(mention_cue, text_snippet_full, window_size, masking=False, index=0):
    ''' input: window size as k means that the k words before and after the mention are included;
               masking, whether to mask the mention itself to [MASK]
               index, the index of the data that being processed with this function, this number has no effect to the output
        output: texts in context window, start char position of the mention in the window, end char position of the mention in the window
    '''
    
    list_mention_tokens = mention_cue.split()
    mention_first_token = list_mention_tokens[0]
    #print(mention_first_token)
    len_mention_token = len(list_mention_tokens)
    #get mention position
    list_tokens = text_snippet_full.split()
    #print('text_snippet_full:',text_snippet_full)
    if '*****' in text_snippet_full:
        #match the first token is enough
        list_positions = [ind for ind, token in enumerate(list_tokens) if '*****'+mention_first_token.lower() in token.lower()]        
    else:
        #match the full mention tokens to the alphanumerised tokens in the sentence
        list_tokens_alphanum = [re.sub('[^A-Za-z0-9]+', '',token.lower()) for token in list_tokens]
        print(list_mention_tokens, list_tokens_alphanum)
        list_positions = [find_sub_list(list_mention_tokens,list_tokens_alphanum)[0]] #only get the first occurrence of the match
        #list_positions = [ind for ind, token in enumerate(list_tokens) if mention_first_token.lower() == re.sub('[^A-Za-z0-9]+', '',token.lower())] #removed all non-alphanumerical characters using regular expressions, like dot(.).
        #print(list_positions, mention_first_token.lower(), re.sub('[^A-Za-z0-9]+', '',token.lower()))
        #list_positions = list_positions[0:1] # get only the first occurrence
    # assert len(list_positions) == 1
    if len(list_positions) != 1:
        print('index %s from get_context_window(): irregular mention cue frequency (as %s) detected!' % (str(index), str(len(list_positions))))
        #print('index %s:' % str(index), 'mention_cue:',mention_cue, '\n', 'text_snippet_full:', text_snippet_full)
        # this irregularity can only happen under the 'if '*****' in text_snippet_full' if-control
        # if len(list_positions) is 0, this means that the mention is not in the text - an issue of SemEHR output regarding the offsets, could be because of documents not fully downloaded to .json files (see how ***** was added in step0_mimic3_data_processing.py).
        return '',0,0
    start_pos = list_positions[0]
    end_pos = list_positions[0] + len_mention_token-1
    #retrive the substring from the split token list
    left_window_size = start_pos if window_size>=start_pos else window_size
    if masking:
        # mask the mention to [MASK]
        text_snippet_context_window = ' '.join(list_tokens[start_pos-left_window_size:start_pos] + ['[MASK]'] + list_tokens[end_pos+1:end_pos+window_size+1])
    else:
        text_snippet_context_window = ' '.join(list_tokens[start_pos-left_window_size:end_pos+window_size+1])
        
    #get the beginning and the end of the char position of the mention in the context window
    if '*****' in text_snippet_context_window:
        ment_begin_offset = text_snippet_context_window.find('*****')
        ment_end_offset = ment_begin_offset + len(mention_cue) - 1
        #remove the '*****' sign
        text_snippet_context_window = text_snippet_context_window.replace('*****','')
    else:
        if '[MASK]' in text_snippet_context_window:
            ment_begin_offset = text_snippet_context_window.find('[MASK]')
            ment_end_offset = ment_begin_offset + len('[MASK]') - 1
        else:
            ment_begin_offset = text_snippet_context_window.find(mention_cue)
            ment_end_offset = ment_begin_offset + len(mention_cue) - 1
            
    #return text_snippet_context_window, left_window_size, left_window_size + len_mention_token-1
    return text_snippet_context_window, ment_begin_offset, ment_end_offset

def get_context_window_from_data_list_tuples(data_list_tuples,window_size=5,masking=False):
    #get context window, and insert the mention offsets in to the new data_list_tuples
    #for parameters see the function above get_context_window()
    #data format updated in return: a 6-element tuple of context window, start_mention_offset, end_mention_offset, document_structure_name, mention_name, label (True or False)
    data_list_tuples_cw = []
    for index, data_tuple in enumerate(data_list_tuples):
        text_snippet_context_window, ment_begin_offset, ment_end_offset = get_context_window(data_tuple[2].lower(), data_tuple[0], window_size=window_size, masking=masking, index=index)
        doc_struc_name = data_tuple[1]
        mention_name = data_tuple[2]
        label = data_tuple[5]
        data_list_tuples_cw.append((text_snippet_context_window, ment_begin_offset, ment_end_offset, doc_struc_name, mention_name, label))
    return data_list_tuples_cw

#unused    
def get_char_offset(context_window, left_window_size, right_window_size):
    ''' get the char offset of the mention in the context window, based on the output of the get_context_window() above
    input: context_window, left_window_size, right_window_size
    output: mention_begin_offset, mention_end_offset'''
    
    ment_begin_offset = 0
    ment_end_offset = 0
    tokens_cw = context_window.split()
    for ind, token in enumerate(tokens_cw):
        if ind < left_window_size:
            ment_begin_offset = ment_begin_offset + len(token) + 1 # also add the space
        if ind <= right_window_size:
            ment_end_offset = ment_end_offset + len(token) + 1
    ment_end_offset = ment_end_offset - 1 - 1
    return ment_begin_offset, ment_end_offset
    
def get_max_length_batch(list_sent):
    return max([len(sent.split()) for sent in list_sent])

def get_max_len_batch_sent_with_pos_tuple(list_sent_retrieved_with_pos_tuple):
    return max([len(sent_with_pos_tuple[0].split()) for sent_with_pos_tuple in list_sent_retrieved_with_pos_tuple])

def find_sub_list(sl,l):
    #'''find the first occurrence of a sublist in list: #from https://stackoverflow.com/a/17870684'''
    #if sublist not found, None is returned.
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1
            
# unused
def TSNE_plot_from_emb(nparray2d_embedding,list_object_label,figname2save='sent_emb.png'):
    #plot 2 dimensional embedding from 2d np array, each row is an embedding (or representation) of an object (word/sentence).
    '''input: 2d nparray embedding, 
              list of label or deescription of each row in the embedding, 
              the figure name to save
    '''          
    emb_norm_2D = TSNE(n_components=2, init='pca', random_state=100).fit_transform(nparray2d_embedding)
    #print(emb_norm_2D)
    plt.scatter(emb_norm_2D[:,0],emb_norm_2D[:,1])
    
    num_objects = nparray2d_embedding.shape[0]
    
    for i in range(num_objects):
        plt.annotate(list_object_label[i],(emb_norm_2D[:,0][i],emb_norm_2D[:,1][i]))
    plt.savefig(figname2save)
    plt.show()
    plt.clf()

def TSNE_plot_from_emb_text_adjusted(nparray2d_embedding,list_object_label,use_pca=True,figname2save='sent_emb.png',fontsize=10,len_sent_anomaly_appended_trail=0):
    #plot 2 dimensional embedding from 2d np array, each row is an embedding (or representation) of an object (word/sentence).
    '''input: 2d nparray embedding, 
              list of label or deescription of each row in the embedding, 
              the figure name to save
              fontsize
              len_sent_anomaly_appended_trail, the number anomaly sentences appended in the end
              
       a reference tutorial on PCA before TSNE: https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    '''          
    
    # first, if chosen, use PCA to reduce the dimensionality from 768/1024/... to 50
    num_objects = nparray2d_embedding.shape[0]
    emb_size = nparray2d_embedding.shape[1]
    if use_pca:
        pca_50 = PCA(n_components=min(num_objects, emb_size, 50))
        nparray2d_embedding = pca_50.fit_transform(nparray2d_embedding)
    
    # second, use TSNE with a fixed random_state for reproducible results
    emb_norm_2D = TSNE(n_components=2, random_state=100).fit_transform(nparray2d_embedding)
    #print(emb_norm_2D)
    
    assert num_objects == len(list_object_label) # test whether the object label list has the same length as the embedding rows
    
    df = pd.DataFrame(emb_norm_2D, index=list_object_label, columns=['x', 'y'])
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(df['x'], df['y'], s=50)

    texts = [ax.text(df['x'][i], df['y'][i], list_object_label[i], ha='center', va='center', fontsize=fontsize) if i+len_sent_anomaly_appended_trail<=num_objects-1 else ax.text(df['x'][i], df['y'][i], list_object_label[i], ha='center', va='center', fontsize=fontsize, color='red') for i in range(num_objects) ] #for the anomaly sentences, set the color to red in text annotation
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))
    plt.savefig(figname2save,dpi=300)
    plt.show()

def encode_data_tuple(data_list_tuples, masking=False , model_path='./NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12/', with_doc_struc=False, marking_str='testing_50', window_size=5, masking_rate=1, diverse_sampling=False, num_of_data_per_mention=10,store_encoding_pik=True,store_cw_encoding_and_men_tokens_dicts=True,start_and_shut_bert_server=True,load_from_data_feature_file=True,port_number_str='5555'):
    '''get BERT-based mention rep from a list of data tuples
    input: (i) data_list_tuple, a 6-element tuple of section_text, document_structure_name, mention_name, UMLS_code, UMLS_desc, label (True or False)
        (ii) masking, whether to mask the mention itself to [MASK]
        (iii) model_path, the path of the tensorflow BERT model
          
        #set model path from one below

        #normal BERT base
        #model_path = './uncased_L-12_H-768_A-12\\uncased_L-12_H-768_A-12\\'

        #blueBERT or NCBI-BERT base and large
        model_path = './NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12/'
        #model_path = './NCBI_BERT_pubmed_mimic_uncased_L-24_H-1024_A-16/'
        #pubmed abstrat+fulltext BERT base (not large version available)
        #model_path = './pubmedBERT_tf_model/'
        (iv) with_doc_struc, whether using document structure
        (v) marking_str, a post-fix for making the feature file name unique.
        (vi) window_size, the size of each side of the context window
        (vii) masking_rate, the percentage of randomly sampled data where their mentions being masked, default as 1 (all masking)
        (viii) diverse_sampling, whether to sample the data diversely, i.e equal number of the mentions.
        (viv) num_of_data_per_mention, this is used only when diverse_sampling is True, this will sample num_of_data_per_mention of data for each mention.
        (vv) store_encoding_pik, whether or not to store the encoding as .pik file, usually set as True, except for demo running and testing each time with new data.
        (vvi) store_cw_encoding_and_men_tokens_dicts, whether or not to store the 2 dicts: 
            (1) dict from context window to the 2-element tuple of its vector representation and the list of tokens.
            (2) dict from mention to the list of Wordpiece tokens, tokenised by using vocab.txt in the BERT model.
        (vvii) start_and_shut_bert_server, whether to start and shut the bert server with python code, if chosen as False, start and shut the server with commanline before and after calling this function
            To start the server:
                bert-serving-start -pooling_strategy NONE -show_tokens_to_client -max_seq_len NONE -model_dir [PATH]
            To shut the server:
                bert-serving-terminate -port 5555
        (vviii) load_from_data_feature_file, whether or not to load encoding features directly from the feature file        
        (vviv) port_number_str, port number as string for bert-as-service, change this if there is a "zmq.error.ZMQError: Address already in use" error
    output: a 4-element tuple, X, y, X as a numpy array of vector representations, y as a numpy array of labels
            list_ind_empty_cw, list of indexes of data of empty context windows
            list_ind_wrong_find_mt, list of indexes of data of wrong find mention tokens    '''
            
    #0. load from pickle file if exists, otherwise start loading data and encoding (also load the dict of encoding if exists)
    model_name = model_path.split('\\')[-2] if '\\' in model_path else model_path.split('/')[-2]
    if masking:
        masking_str = '_[MASK]' if masking_rate == 1 else '_[MASK]%s' % str(masking_rate)
    else:
        masking_str = ''
        
    data_feature_file_name = 'mention_disamb_data_ft%s_%s_ws%s%s_%s%s.pik' % (masking_str, model_name, str(window_size), '_ds' if with_doc_struc else '', marking_str, '_divs%s' % str(num_of_data_per_mention) if diverse_sampling else '')
    print('checking:', data_feature_file_name)
    if os.path.exists(data_feature_file_name) and load_from_data_feature_file:
        with open(data_feature_file_name, 'rb') as data_f:
            output_tuple = pickle.load(data_f)
            if len(output_tuple) == 2:
                X, y = output_tuple
                return X, y
            else:
                X, y, list_ind_empty_cw, list_ind_wrong_find_mt = output_tuple
                return X, y, list_ind_empty_cw, list_ind_wrong_find_mt
    
    dict_sent2rep_with_tokens_file_name = 'dict_sent2rep%s_%s_ws%s%s.pik' % (masking_str, model_name, str(window_size), '_ds' if with_doc_struc else '') # no marking_str or sampling marks. This dict is used to store the representation of all context windows.
    if os.path.exists(dict_sent2rep_with_tokens_file_name):
        with open(dict_sent2rep_with_tokens_file_name, 'rb') as data_f:
            dict_sent2rep_with_tokens = pickle.load(data_f)
            print('dict_sent2rep_with_tokens found', type(dict_sent2rep_with_tokens), len(dict_sent2rep_with_tokens))
    else:
        dict_sent2rep_with_tokens = {} # store the processed sentences and their rep and BERT tokens
    
    dict_mention_list_tokens_file_name = 'dict_ment2tokens%s_%s.pik' % (masking_str, model_name) # no marking_str or sampling marks. This dict is used to store the representation of all context windows.
    if os.path.exists(dict_mention_list_tokens_file_name):
        with open(dict_mention_list_tokens_file_name, 'rb') as data_f:
            dict_mention_list_tokens = pickle.load(data_f)
            print('dict_mention_list_tokens found', type(dict_mention_list_tokens), len(dict_mention_list_tokens))
    else:
        dict_mention_list_tokens = {} # store the processed sentences and their rep and BERT tokens
        
    for i, key in enumerate(dict_sent2rep_with_tokens.keys()): 
        if i<10:
            print(key)
    
    for i, key in enumerate(dict_mention_list_tokens.keys()): 
        if i<10:
            print(key)
            
    #1. update the data_list_tuples with the context windows for the sentences
    #data format updated: a 4-element tuple of (i) context window, (ii) document_structure_name, (iii) mention_name, (iv) label (True or False)
    num_of_data = len(data_list_tuples)
    data_list_tuples_cw = [(get_context_window(data_tuple[2].lower(), data_tuple[0], window_size, masking=True if masking and ind<num_of_data*masking_rate else False, index=ind)[0],data_tuple[1],'[MASK]' if masking and ind<num_of_data*masking_rate else data_tuple[2],data_tuple[5]) for ind, data_tuple in enumerate(data_list_tuples)] #masking for each data: if masking as true and within the masking_rate; the mention_name is also '[MASK]'ed.
    #print(data_list_tuples[0:2])
    print(data_list_tuples_cw[0:2])
    assert len(data_list_tuples_cw) == len(data_list_tuples) # the formatted data tuple should be of the same data size of the original data tuple
    
    #print('num of instances of wrong ***** mark in data_list_tuples:', list_sent_cont_window.count(''), 'out of', len(list_sent_cont_window))
    #there are 80 wrong ***** marks in data_list_tuples, these instances are to be dropped.
    
    #diverse sampling    
    if diverse_sampling:
        data_list_tuples_cw_sampled = []
        dict_mention_freq = {}
        for i, data_tuple_cw in enumerate(data_list_tuples_cw):
            #mention = data_tuple_cw[2]
            mention = data_list_tuples[i][2] # get the mention from the original data tuple (as in the formatted tuple, the mention can be already [MASK]-ed
            if dict_mention_freq.get(mention,None) == None:
                dict_mention_freq[mention] = 1
                data_list_tuples_cw_sampled.append(data_tuple_cw)
            else:
                dict_mention_freq[mention] += 1
                if dict_mention_freq[mention]<=num_of_data_per_mention:
                    #only append the data tuple when the mention appears less or equal to certain times
                    data_list_tuples_cw_sampled.append(data_tuple_cw)                    
        # cover the full data_list_tuples_cw with the sampled ones
        data_list_tuples_cw = data_list_tuples_cw_sampled
        
    #2. get X and y numpy arrays
    # list_sent_cw = [data_tuple_cw[0] for data_tuple_cw in data_list_tuples_cw if data_tuple_cw[0] != '' ] # this will eliminate entries that have empty context window due to 'irregular mention cue frequency' detected in the get_context_window() function. 
    # list_doc_struc = [data_tuple_cw[1] for data_tuple_cw in data_list_tuples_cw if data_tuple_cw[0] != '' ]
    # list_mentions = [data_tuple_cw[2] for data_tuple_cw in data_list_tuples_cw if data_tuple_cw[0] != '' ]
    # list_bi_labels = [data_tuple_cw[3] for data_tuple_cw in data_list_tuples_cw if data_tuple_cw[0] != '' ]
    #print('original', len(data_list_tuples_cw), 'after eliminating empty sents', len(list_sent_cw))
    list_sent_cw = [data_tuple_cw[0] for data_tuple_cw in data_list_tuples_cw]; assert len(list_sent_cw) == len(data_list_tuples_cw) #list_sent_cw can have empty string elements due to 'irregular mention cue frequency' detected in the get_context_window() function. #ensuring that the number of data entries is preserved.
    list_doc_struc = [data_tuple_cw[1] for data_tuple_cw in data_list_tuples_cw]
    list_mentions = [data_tuple_cw[2] for data_tuple_cw in data_list_tuples_cw]
    list_bi_labels = [data_tuple_cw[3] for data_tuple_cw in data_list_tuples_cw]
    
    print(list_sent_cw[0:5])
    print(list_bi_labels[0:5])
    print('pos:', list_bi_labels.count(1), 'neg:', list_bi_labels.count(0))
    
    # intialise the huggingface BERTWordPiece tokenizer (with the vocab.txt file)
    #vocab_filename = "%s\\vocab.txt" % model_path
    vocab_filename = os.path.join(model_path,'vocab.txt')
    print('vocab_file:',vocab_filename)
    WordPiece = BertWordPieceTokenizer(vocab_filename)
    # to learn more about tokenisation with the Huggingface library: https://heartbeat.fritz.ai/hands-on-with-hugging-faces-new-tokenizers-library-baff35d7b465
    
    
    #BERT encoding to get X
    if start_and_shut_bert_server:
        #BERT server start

        # to concatenate the last 4 layers: '-pooling_layer', '-4','-3','-2','-1',
        args = get_args_parser().parse_args(['-model_dir', model_path,
                                             '-pooling_strategy', 'NONE',
                                             '-pooling_layer', '-2',
                                             '-max_seq_len', 'None',  
                                             '-max_batch_size','256',
                                             '-show_tokens_to_client',
                                             '-graph_tmp_dir','./graph_tmp/',
                                             '-port',port_number_str])
        server = BertServer(args)
        server.start()
    #also: to start the server with commandline: bert-serving-start -pooling_strategy NONE -show_tokens_to_client -max_seq_len NONE -model_dir .\NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12
    
    # BERT client start
    from bert_serving.client import BertClient    
    bc = BertClient()
    
    # get the mention rep for each sentence and stack them to feature matrix
    #dict_mention_list_tokens = {} # initialised previously from file, store the processed mentions and their BERT tokens    
    #dict_sent2rep_with_tokens = {} # initialised previously from file, store the processed sentences and their rep and BERT tokens
    list_ind_empty_cw = [] # store the index of the case the context window is empty (due to 'irregular mention cue frequency' detected in the get_context_window() function)
    list_ind_wrong_find_mt = [] # store the index of the case that the mention tokens not included in the setence tokens
    for ind, sent_cw in enumerate(tqdm(list_sent_cw)):
    #for ind, sent_cw in enumerate(list_sent_cw):
        
        if sent_cw == '':
            list_ind_empty_cw.append(ind) # storing the erroneous ones due to mention not appearing in the text
            continue
        # get tokenised mention; using a dict to store tokens of the processed mentions for fast speed.
        mention = list_mentions[ind] # here can be [MASK] in the list_mentions
        #print(mention)
        if dict_mention_list_tokens.get(mention, None) == None:
            #_,list_tokens_mention = bc.encode([mention],show_tokens=True) # through encoding: this cannot correctly tokenize '[MASK]'        
            list_tokens_mention = WordPiece.encode(mention).tokens # with the huggingface BERTWordPiece tokeniser
            dict_mention_list_tokens[mention] = list_tokens_mention
        else:
            list_tokens_mention = dict_mention_list_tokens[mention]
                              
        #mention_cue_tokenised = list_tokens_mention[0][1:-1] # not including the [CLS] at the beginning and the [SEP] at the end. # if using bc.encode to get tokenised sentence
        mention_cue_tokenised = list_tokens_mention[1:-1] # not including the [CLS] at the beginning and the [SEP] at the end. if using WordPiece for tokenisation.
        #print(list_tokens_mention,mention_cue_tokenised)

        #encoding the whole sentence and every token in it; using a dict to store rep of the processed sentences for fast speed.
        #sent_cw = list_doc_struc[ind] + ' ||| ' + sent_cw if with_doc_struc else sent_cw # add the document structure as the first part of the sentences if chosen to; ' ||| ' means the sentence separation [SEP].
        #if dict_sent2rep_with_tokens.get(sent_cw, None) == None:
        if not sent_cw in dict_sent2rep_with_tokens:
            print('start encoding %s' % sent_cw)
            #vec,list_tokens_sent = bc.encode([sent_cw],show_tokens=True)
            sent_cw_tokenised = WordPiece.encode(sent_cw).tokens[1:-1]  
            if with_doc_struc:
                doc_struc_tokenised = WordPiece.encode(list_doc_struc[ind]).tokens[1:-1]
                sent_cw_tokenised = doc_struc_tokenised + ['|||'] + sent_cw_tokenised
            vec,list_tokens_sent = bc.encode([sent_cw_tokenised],show_tokens=True,is_tokenized=True)
            dict_sent2rep_with_tokens[sent_cw] = (vec,list_tokens_sent)
        else:
            vec,list_tokens_sent = dict_sent2rep_with_tokens[sent_cw]
        #vec,list_tokens_sent = vec_all_sent[ind:ind+1],list_tokens_all_sents[ind:ind+1]
        #print(vec,list_tokens_sent)
        if ind < 3:
            print(sent_cw,list_tokens_sent) #just display the first three
        
        # get the index of the start and end of the wordpiece of the mention in the list of tokens in the sentence.
        list_start_end_inds_tuple = [find_sub_list(mention_cue_tokenised,list_tokens) if find_sub_list(mention_cue_tokenised,list_tokens)!=None else find_sub_list(['##' + mc_part if mc_part[:2] != '##' else mc_part for mc_part in mention_cue_tokenised],list_tokens) for list_tokens in list_tokens_sent] # find the mention tokens in the sentence tokens, if not there then '##'-ise the mention tokens.
        #print(list_start_end_inds_tuple)
        if list_start_end_inds_tuple == [None]: # ignore the instance of the mention tokens are not included in the sentence rep
            list_ind_wrong_find_mt.append(ind)
            output_to_file('wrong_find_mention_token_%s.txt' % str(ind),str((mention_cue_tokenised,list_tokens_sent)))
            continue
        # get the vectors of the wordpiece in the mention: output a list of 2D vectors
        list_nparray2d_mention_wordpiece_reps_in_sent = [vec[i][start_end_inds_tuple[0]:start_end_inds_tuple[1]+1] for i, start_end_inds_tuple in enumerate(list_start_end_inds_tuple)]
        nparray3d_mention_wordpiece_rep_in_sent = np.array(list_nparray2d_mention_wordpiece_reps_in_sent) # turn to 3D vectors
        #print(nparray3d_mention_wordpiece_rep_in_sent.shape,nparray3d_mention_wordpiece_rep_in_sent)
        # the contextual mention representation is the (i) average or (ii) the first subword of the wordpiece reprsentations
        # (i) averaging
        nparray2d_mention_rep_in_sent = np.mean(nparray3d_mention_wordpiece_rep_in_sent,axis=1)
        # (ii) first subword
        #nparray2d_mention_rep_in_sent = nparray3d_mention_wordpiece_rep_in_sent[:,0,:]
        #print(nparray2d_mention_rep_in_sent.shape, nparray2d_mention_rep_in_sent[:,:5]) # shape (num_sent,hidden_size) and show the first five dimension for each mention rep.
        if ind == 0:
            X = nparray2d_mention_rep_in_sent
        else:
            X = np.append(X,nparray2d_mention_rep_in_sent,axis=0)
            #print(X.shape)
        #print(nparray2d_mention_rep_in_sent)
        
    #turn list_bi_labels (list of boolean) to nparray of 0/1s
    list_bi_labels = [bi_label for ind, bi_label in enumerate(list_bi_labels) if (ind not in list_ind_empty_cw) and (ind not in list_ind_wrong_find_mt)] # filter out the problematic data entries: (i) empty context window (ii) mention tokens not in sentence tokens
    y = np.array(list_bi_labels).astype(int)
    
    if store_encoding_pik:
        #store the data (as a 2-element tuple) to a .pik file
        with open(data_feature_file_name, 'wb') as data_f:
            pickle.dump((X, y, list_ind_empty_cw, list_ind_wrong_find_mt), data_f)        
        
        print('data stored to',data_feature_file_name)
    
    if store_cw_encoding_and_men_tokens_dicts:
        with open(dict_sent2rep_with_tokens_file_name, 'wb') as data_f:
            pickle.dump(dict_sent2rep_with_tokens, data_f)        
        
        print('dictionary of context window encoding and tokens stored to',dict_sent2rep_with_tokens_file_name)
        
        with open(dict_mention_list_tokens_file_name, 'wb') as data_f:
            pickle.dump(dict_mention_list_tokens, data_f)        
        
        print('dictionary of mention to list of tokens stored to',dict_mention_list_tokens_file_name)
        
    if start_and_shut_bert_server:
        #3. BERT server shut
        shut_args = get_shutdown_parser().parse_args(['-ip','localhost',
                                                      '-port',port_number_str,
                                                      '-timeout','5000'])
        BertServer.shutdown(shut_args)    
    #also: to shut down the server with commandline: bert-serving-terminate -port 5555
    
    return X,y, list_ind_empty_cw, list_ind_wrong_find_mt
        
# using average word2vec embedding of context window to encode the mention
def encode_data_tuple_ave_word_emb(data_list_tuples, word2vec_model_path='processed_full.w2v', masking=False , with_doc_struc=False, marking_str='testing_50', window_size=5, store_encoding_pik=True):    
    '''get w2v averaged rep from a list of data tuples
    input: (i) data_list_tuple, a 6-element tuple of section_text, document_structure_name, mention_name, UMLS_code, UMLS_desc, label (True or False)
        (ii) word2vec_model_path, gensim word2vec model path
        (iii) masking, whether masking the mention
        (iv) with_doc_struc, whether using document structure
        (v) marking_str, a post-fix for making the feature file name unique.
        (vi) window_size, the size of each side of the context window
        (vii) store_encoding_pik, whether or not to store the encoding as .pik file, usually set as True, except for demo running and testing each time with new data.
        
    output: a 3-element tuple, X, y, X as a numpy array of vector representations, y as a numpy array of labels
            list_ind_empty_cw, list of indexes of data of empty context windows
    '''
    #0. load from pickle file if exists, otherwise start loading data and encoding
    data_feature_file_name = 'mention_disamb_data_w2v_ft_ws%s%s%s_%s.pik' % (str(window_size), '_masked' if masking else '', '_ds' if with_doc_struc else '', marking_str)
    if os.path.exists(data_feature_file_name):
        with open(data_feature_file_name, 'rb') as data_f:
            output_tuple = pickle.load(data_f)
            X, y, list_ind_empty_cw = output_tuple
            return X, y, list_ind_empty_cw
                
    #1. update the data_list_tuples with the context windows for the sentences
    #data format updated: a 4-element tuple of (i) context window, (ii) document_structure_name, (iii) mention_name, (iv) label (True or False)
    num_of_data = len(data_list_tuples)
    data_list_tuples_cw = [(get_context_window(data_tuple[2].lower(), data_tuple[0], window_size, masking=True if masking else False, index=ind)[0],data_tuple[1],'[MASK]' if masking else data_tuple[2],data_tuple[5]) for ind, data_tuple in enumerate(data_list_tuples)] #masking for each data: if masking as true and within the masking_rate; the mention_name is also '[MASK]'ed.
    #print(data_list_tuples[0:2])
    print(data_list_tuples_cw[0:2])
    assert len(data_list_tuples_cw) == len(data_list_tuples) # the formatted data tuple should be of the same data size of the original data tuple
    
    #print('num of instances of wrong ***** mark in data_list_tuples:', list_sent_cont_window.count(''), 'out of', len(list_sent_cont_window))
    #there are 80 wrong ***** marks in data_list_tuples, these instances are to be dropped.
    
    #2. get X and y numpy arrays
    list_sent_cw = [data_tuple_cw[0] for data_tuple_cw in data_list_tuples_cw]; assert len(list_sent_cw) == len(data_list_tuples_cw) #list_sent_cw can have empty string elements due to 'irregular mention cue frequency' detected in the get_context_window() function. #ensuring that the number of data entries is preserved.
    list_doc_struc = [data_tuple_cw[1] for data_tuple_cw in data_list_tuples_cw]
    list_mentions = [data_tuple_cw[2] for data_tuple_cw in data_list_tuples_cw]
    list_bi_labels = [data_tuple_cw[3] for data_tuple_cw in data_list_tuples_cw]
    
    print(list_sent_cw[0:5])
    print(list_bi_labels[0:5])
    print('pos:', list_bi_labels.count(1), 'neg:', list_bi_labels.count(0))
    
    # get the mention rep for each sentence and stack them to feature matrix
    dict_sent2rep = {} # store the processed sentences and their rep
    list_ind_empty_cw = [] # store the index of the case the context window is empty (due to 'irregular mention cue frequency' detected in the get_context_window() function)
    
    # load gensim word2vec model
    if word2vec_model_path.endswith('.txt'):
        # a .txt file of word2vec format, see https://radimrehurek.com/gensim/scripts/glove2word2vec.html
        #tmp_file = get_tmpfile(word2vec_model_path)
        model = KeyedVectors.load_word2vec_format(word2vec_model_path)
    else: # a pre-trained word2vec model file
        model = Word2Vec.load(word2vec_model_path)
    for ind, sent_cw in enumerate(tqdm(list_sent_cw)):
        if sent_cw == '':
            list_ind_empty_cw.append(ind)
            continue
        
        #print(sent_cw)
        if dict_sent2rep.get(sent_cw, None) is None:
        #if not sent_cw in dict_sent2rep:
            if with_doc_struc:
                sent_cw = list_doc_struc[ind].replace('_',' ') + ' ' + sent_cw #here split the words in the document structure name, e.g. "hospital_discharge_physical" to "hospital discharge physical", so that the words can be found in the pre-trained w2v embedding vocabularies. 
            vec = encode_one_sent_ave_word_emb(sent_cw,model)
            dict_sent2rep[sent_cw] = vec
        else:
            vec = dict_sent2rep[sent_cw]
        
        vec_2d = np.expand_dims(vec, axis=0)
        if ind == 0:
            X = vec_2d
        else:
            X = np.append(X,vec_2d,axis=0)
            #print(X.shape)
            
    #turn list_bi_labels (list of boolean) to nparray of 0/1s
    list_bi_labels = [bi_label for ind, bi_label in enumerate(list_bi_labels) if ind not in list_ind_empty_cw] # filter out the problematic data entries: (i) empty context window (ii) mention tokens not in sentence tokens
    y = np.array(list_bi_labels).astype(int)
    
    if store_encoding_pik:
        #store the data (as a 2-element tuple) to a .pik file
        with open(data_feature_file_name, 'ab') as data_f:
            pickle.dump((X, y, list_ind_empty_cw), data_f)        
        
        print('data stored to',data_feature_file_name)
    
    return X,y, list_ind_empty_cw
    
def encode_one_sent_ave_word_emb(sent,gensim_w2v_model):
    #encode one sentence with average word embedding
    #if embedding not found for the token, do not count it; return an all-zero vector when no tokens in the sentence have an embedding.
    
    #remove the '[MASK]' token.
    sent = sent.replace(' [MASK] ',' ')
    #tokenisation same as in the original CAML-MIMIC step for tokens in each sentence
    #list_tokens = sent.split(' ')
    tokenizer = RegexpTokenizer(r'\w+')
    list_tokens = [t.lower() for t in tokenizer.tokenize(sent) if not t.isnumeric()]
    
    word_vectors = gensim_w2v_model.wv
    rep = np.zeros(len(word_vectors[word_vectors.index2word[0]]))
    len_in_calculation = len(list_tokens)
    for token in list_tokens:
        token = token.lower() # the embedding was trained on lower cased texts.
        if token in word_vectors.vocab:
            rep = rep + gensim_w2v_model.wv[token]
        else:
            print(token + ' embedding not exist')
            #if embedding not found for the token, do not count it
            len_in_calculation = len_in_calculation - 1            
    rep = rep / len_in_calculation if len_in_calculation != 0 else np.zeros(len(word_vectors[word_vectors.index2word[0]])) #return an all-zero vector when no tokens in the sentence have an embedding.
    return rep
    
# input the encoding output_tuple
# output the trained model
# test_size portion \in (0,1) default as 0.1, not outputting validation results if test_size as 0.
# balanced sampling default as False
# regularisation penalty C default as 1
def get_model_from_encoding_output(output_tuple, num_of_training_samples, test_size=0.1, balanced_sampling=False, C=1,report_binary_class_dist_output=False, report_train_validation_results=False):
    if len(output_tuple) == 2:
        X, y = output_tuple
        list_ind_empty_cw, list_ind_wrong_find_mt = [], []
    elif len(output_tuple) == 3:
        #this is for the output of encode_data_tuple_ave_word_emb()
        X, y, list_ind_empty_cw = output_tuple
        list_ind_wrong_find_mt = []
    else:
        #this is for the output of encode_data_tuple()
        X, y, list_ind_empty_cw, list_ind_wrong_find_mt = output_tuple
    # X, y = load_data('mention_disamb_data_features_NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12_sample_89902.pik')
    #print('number of entries with issues: %s+%s' % str(len(list_ind_empty_cw)), str(len(list_ind_wrong_find_mt)))
    #print(list_ind_wrong_find_mt)
    # if not masked_training:
        # #remove the list of index there were eliminated during the wrong_find_mention_token process
        # list_ind_wrong_find_mt = [26964,41622,51767,65705,67399,85607,87693]
        # y = np.array([ele for ind, ele in enumerate(y.tolist()) if ind not in list_ind_wrong_find_mt])
    
    print('encoding done')
    #print(y.shape, type(y), len(y.tolist()))
    print('X,y:', X.shape, type(X), y.shape, type(y))
    print(report_binary_class_dist(y))
    
    #data standardisation
    #X = preprocessing.scale(X)

    #data split
    if test_size != 0:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=1234)
    else:
        # if test size not as 0, then do not do data split
        X_train, y_train = X, y
        
    training_data_len = X_train.shape[0]
    print(training_data_len)
    
    if not balanced_sampling:
        #adjust the number of training samples by the error samples
        #this ensure the different training settings will use the exact same set of training data (while adjusted numbers may be slightly different)
        num_of_training_samples_adjusted = num_of_training_samples
        for ind in list_ind_empty_cw + list_ind_wrong_find_mt:
            if ind < num_of_training_samples:
                num_of_training_samples_adjusted = num_of_training_samples_adjusted - 1            
        #try only a random sample of the whole training data: the first num_of_training_samples (adjusted by the error samples), X_train is the randomly shuffled data.
        X_train = X_train[:num_of_training_samples_adjusted,:]
        y_train = y_train[:num_of_training_samples_adjusted]
        print(X_train.shape, type(X_train), type(y_train))
    else:
        #sample a label-balanced set from X_train and Y_train
        #and the number of the whole set is equal to num_of_training_samples
        num_tr_samples_per_label = round(num_of_training_samples/2)
        #print(type(num_tr_samples_per_label), num_tr_samples_per_label)
        data_ind_neg = np.where(y_train == 0)[0]
        data_ind_pos = np.where(y_train == 1)[0]
        num_tr_sample_minority_label = min(len(data_ind_pos),len(data_ind_neg))
        if num_tr_sample_minority_label < num_tr_samples_per_label:
            num_tr_samples_per_label = num_tr_sample_minority_label # adjust the number of training samples per label if it is more than the actual number of data for the minority label (here should be the positive label)
        data_ind_balanced_sample = np.sort(np.concatenate([data_ind_pos[:num_tr_samples_per_label], data_ind_neg[:num_tr_samples_per_label]]))
        X_train = X_train[data_ind_balanced_sample]
        y_train = y_train[data_ind_balanced_sample]
        print(X_train.shape, type(X_train), type(y_train))
    
    #report binary class distribution statistics
    #_, counts = np.unique(y_train, return_counts=True)
    #print('training data pos:', counts[1], 'neg:', counts[0])
    binary_class_dist_str = report_binary_class_dist(y_train)
    print(binary_class_dist_str)
    #if (not (1 in y_train)) or (not (0 in y_train)):
    #    #if there is only one class or no classes
    #    print('Error: only one class in the training data')
    #    return None
        
    #train
    #clf = SVC(kernel="linear", C=0.025)
    clf = LogisticRegression(C=C, penalty='l2') #solver='liblinear' #max_iter=500
    #clf = LinearRegression()
    #clf = MLPClassifier(hidden_layer_sizes=10)
    clf.fit(X_train, y_train)
    #get training results
    y_pred_train = clf.predict(X_train)
    training_results_report = 'training precision: %s' % str(precision_score(y_pred_train, y_train)) +  ' recall: %s' % str(recall_score(y_pred_train, y_train)) + ' F1: %s' % str(f1_score(y_pred_train, y_train))
    print(training_results_report)

    validation_results_report = ''
    if test_size != 0:
        #get validation results
        y_pred = clf.predict(X_valid)
        print(y_pred, type(y_pred))
        #y_pred = y_pred.round() # for linear regression
        #print(y_pred, y_valid)
        validation_results_report = '(weak) validation precision: %s' % precision_score(y_pred, y_valid) + ' recall: %s' % recall_score(y_pred, y_valid) + ' F1: %s' % f1_score(y_pred, y_valid)
        print(validation_results_report)
    
    train_and_valid_results_report = training_results_report + '\n' + validation_results_report
    
    if not report_binary_class_dist_output:
        if not report_train_validation_results:
            return clf
        else:
            return clf, train_and_valid_results_report
    else:    
        if not report_train_validation_results:
            return clf, binary_class_dist_str
        else:
            return clf, binary_class_dist_str, train_and_valid_results_report
            
# test a model and output the predictions for final metric calculation    
# input:(i) the encoding output_tuple
#       (ii) number of first k samples to test
#       (iii) the most to test    
# output: (i) the gold labels
#         (ii) the predictions
#         (iii) the list of erroneous data in the test set when encoding
def test_model_from_encoding_output(output_tuple_test, num_of_testing_samples, clf):    
    if len(output_tuple_test) == 2:
        X_test, y_test = output_tuple_test
        list_ind_empty_cw_test, list_ind_wrong_find_mt_test = [], []
    elif len(output_tuple_test) == 3:
        #this is for the output of encode_data_tuple_ave_word_emb()
        X_test, y_test, list_ind_empty_cw_test= output_tuple_test          
        list_ind_wrong_find_mt_test = []
    else:
        #this is for the output of encode_data_tuple()
        # it also outputs the problematic data entry indexes. we will use these when we fill the results into the spreadsheet.
        X_test, y_test, list_ind_empty_cw_test, list_ind_wrong_find_mt_test = output_tuple_test          
    print('list_ind_empty_cw_test:', list_ind_empty_cw_test)
    print('list_ind_wrong_find_mt_test:', list_ind_wrong_find_mt_test)
    
    list_ind_err_test = list_ind_empty_cw_test + list_ind_wrong_find_mt_test
    list_ind_err_test.sort() # sort the index list from low to high
    #adjust the number of testing samples by the error samples
    num_of_testing_samples_adjusted = num_of_testing_samples
    for ind in list_ind_empty_cw_test + list_ind_wrong_find_mt_test:
        if ind < num_of_testing_samples:
            num_of_testing_samples_adjusted = num_of_testing_samples_adjusted - 1
            
    print('num_of_whole_testing_data:', len(y_test))
    # to note: there might be unlabelled data in y_test, otherwise 0 or 1.
    # get the labelled part in y_test
    y_test_labelled = [y_test[ind] for ind, y_test_ele in enumerate(y_test) if (y_test_ele == 0 or y_test_ele == 1) and ind<num_of_testing_samples_adjusted]
    print('num_of_actual_testing_data:',len(y_test_labelled))
    
    # check the filtered out testing samples
    for ind, y_test_ele in enumerate(y_test):
        if not ((y_test_ele == 0 or y_test_ele == 1)) and ind<num_of_testing_samples_adjusted:
            print('not labelled test sample:',ind,y_test_ele)
            #list_ind_err_test.append(ind) - this is not the original ind, to fix later
    #data standardisation
    #X_test = preprocessing.scale(X_test)

    y_pred_test = clf.predict(X_test)
    #y_pred_test = y_pred_test.round() # for linear regression
    
    #print('model results all %s:' % str(len(y_test_labelled)))
    y_pred_test_labelled = [y_pred_test[ind] for ind, y_test_ele in enumerate(y_test) if  (y_test_ele == 0 or y_test_ele == 1) and ind<num_of_testing_samples_adjusted]
        
    return y_test_labelled, y_pred_test_labelled, list_ind_err_test

#report the binary class/label distribution from the numpy array y (labels of all data)
def report_binary_class_dist(y):
    unique, counts = np.unique(y, return_counts=True)
    data_class_report_str = 'whole data'
    for i, class_type in enumerate(unique):
        if class_type == 1:
            data_class_report_str = data_class_report_str + ' pos: ' + str(counts[i])
        else:
            data_class_report_str = data_class_report_str + ' neg: ' + str(counts[i])
    return data_class_report_str
    
if __name__ == '__main__':
    gensim_w2v_model = Word2Vec.load('processed_full.w2v')
    #print(gensim_w2v_model.wv['ffaewfraewfaewfrawre'])
    vec = encode_one_sent_ave_word_emb('i love you', gensim_w2v_model)
    print(vec,vec.shape)
    
    ##########################################################################
    sys.exit(0) # below are the previous testing code to get contextual BERT embedding of mentions and visualise them. - which does not run unless this line is commented out.
    ##########################################################################
    
    #1. extract all sentences annotated with an UMLS concept
    #set model path, mention name, and window size
    #model_path = 'C:\\Users\\hdong3\\Downloads\\uncased_L-12_H-768_A-12\\'
    model_path = 'C:\\Users\\hdong3\\Downloads\\NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12\\'
    #model_path = 'C:\\Users\\hdong3\\Downloads\\NCBI_BERT_pubmed_mimic_uncased_L-24_H-1024_A-16\\'
    #model_path = 'C:\\Users\\hdong3\\Downloads\\pubmedBERT_tf_model\\'
    model_name = model_path.split('\\')[-2]
    
    add_anomaly_sents = False
    #mention_cue = 'Giant cell arteritis'
    #mention_cue = 'tetanus'; add_anomaly_sents = True
    #mention_cue = 'Asbestosis'
    mention_cue = 'HD'
    
    window_size = 3
    num_of_sent_to_encode = 50
    
    masking = False
    
    #retrive sentences which contain the mention recognised by SemEHR
    df = load_df('df_MIMIC-III DS-Rare-Disease-ICD9-new-rows10000.pik')
    #display column titles
    for col in df.columns: 
        print(col) 
    
    list_sent_retrieved_with_pos_tuple = retrieve_sent_and_doc_structure_from_mention(df, mention_cue, window_size)
    #print(list_sent_retrieved_with_pos_tuple)
    print([(' '.join(sent_with_pos_tuple[0].split()[sent_with_pos_tuple[1]:sent_with_pos_tuple[2]+1]),len(sent_with_pos_tuple[0].split()),sent_with_pos_tuple[0]) for sent_with_pos_tuple in list_sent_retrieved_with_pos_tuple])
    
    max_sent_length = get_max_len_batch_sent_with_pos_tuple(list_sent_retrieved_with_pos_tuple)
    print(max_sent_length)
    
    # intialise the word piece tokenizer with the vocab.txt file
    # Load pre-trained model tokenizer (vocabulary)
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Tokenize the mention with the BERT tokenizer.
    #mention_cue_tokenised = tokenizer.tokenize(mention_cue)
    # Vocabulary filename
    vocab_filename = "%s\\vocab.txt" % model_path
    #vocab_filename = os.path.join()
    # # Instantiate a Bert tokenizers
    WordPiece = BertWordPieceTokenizer(vocab_filename)
    #mention_cue_tokenised = WordPiece.encode(mention_cue)
    # # to learn more about tokenisation with the Huggingface library: https://heartbeat.fritz.ai/hands-on-with-hugging-faces-new-tokenizers-library-baff35d7b465
    #print(mention_cue_tokenised.tokens)
    
    #2. obtain BERT contextualised vectors of the mention from each of the sentences

    # BERT server start
    # to concatenate the last 4 layers: '-pooling_layer', '-4','-3','-2','-1',
    args = get_args_parser().parse_args(['-model_dir', model_path,
                                         '-pooling_strategy', 'NONE',
                                         '-pooling_layer', '-2',
                                         '-max_seq_len', 'None',
                                         
                                         '-cpu',
                                         '-show_tokens_to_client'])
                                         #'-cased_tokenization',
    server = BertServer(args)
    server.start()
    
    # BERT client start
    from bert_serving.client import BertClient    
    bc = BertClient()
    
    # encode mention contextually in sentences
    #get the list of sentences
    list_sent = [sent_with_pos_tuple[0] for sent_with_pos_tuple in list_sent_retrieved_with_pos_tuple]
    #sample after shuffle
    random.Random(1234).shuffle(list_sent)
    list_sent = list_sent[:num_of_sent_to_encode]
    print(list_sent)
    sys.exit(0)
    len_sent_normal = len(list_sent)
    
    #add an anomaly sentence with the same mention
    if add_anomaly_sents:
        list_sent_anomaly = []
        list_sent_anomaly.append('He was diagnosed with tetanus yesterday severely ill.')
        list_sent_anomaly.append('He has tetanus.')
        list_sent_anomaly.append('Past medical history: tetanus.')
        list_sent_anomaly.append('tetanus')
        # get context windows for the anomaly sentences
        list_sent_anomaly_cont_window = [get_context_window(mention_cue.lower(), sent_anomaly, window_size)[0] for sent_anomaly in list_sent_anomaly]
        # append the cont windows to the end of 'normal' sentences
        list_sent = list_sent + list_sent_anomaly_cont_window
    len_sent_anomaly_appended_trail = len(list_sent) - len_sent_normal # get the number of anomaly sentences appended at the end
    
    # masking the mention in the list
    if masking:
        # case insensitive replace of mention_cue to '[MASK]'
        mention_cue_src_str  = re.compile(mention_cue, re.IGNORECASE) 
        list_sent = [mention_cue_src_str.sub('[MASK]', sent) for sent in list_sent]
        mention_cue = '[MASK]'
        
    #get tokenised mention
    # through encoding: this cannot correctly tokenize '[MASK]'
    # _,list_tokens_mention = bc.encode([mention_cue],show_tokens=True)
    # mention_cue_tokenised = list_tokens_mention[0][1:-1] # not including the [CLS] at the beginning and the [SEP] at the end.
    # print(list_tokens_mention,mention_cue_tokenised)
    # through the official tokenizer in huggingface
    list_tokens_mention = WordPiece.encode(mention_cue).tokens
    mention_cue_tokenised = list_tokens_mention[1:-1] # not including the [CLS] at the beginning and the [SEP] at the end.
    print(list_tokens_mention,mention_cue_tokenised)
    
    #encoding the whole sentence and every token in it.
    # option 1: encode without tokenisaiton
    # vec,list_tokens_sent = bc.encode(list_sent,show_tokens=True)
    # print(vec)
    # print('now print tokens')
    # for sent,tokens_sent in zip(list_sent,list_tokens_sent):
        # print(sent,tokens_sent)
        # tokens_sent_ = WordPiece.encode(sent).tokens
        # print('from BERTWordPieceTokenizer:')
        # print(sent,tokens_sent_)
        # assert tokens_sent == tokens_sent_
    
    # option 2: encode with tokenisation
    list_sent_tokenised = [WordPiece.encode(sent).tokens[1:-1] for sent in list_sent] # tokenise the sentences, do not include [CLS] and [SEP] in the BERTWordPiece tokeniser output
    vec,list_tokens_sent = bc.encode(list_sent_tokenised,show_tokens=True,is_tokenized=True)
    for sent,tokens_sent in zip(list_sent,list_tokens_sent):
        print(sent,tokens_sent)
        
    # get the index of the start and end of the wordpiece of the mention in the list of tokens in the sentence.
    list_start_end_inds_tuple = [find_sub_list(mention_cue_tokenised,list_tokens) for list_tokens in list_tokens_sent]
    print(list_start_end_inds_tuple)
    # get the vectors of the wordpiece in the mention: output a list of 2D vectors
    list_nparray2d_mention_wordpiece_reps_in_sent = [vec[i][start_end_inds_tuple[0]:start_end_inds_tuple[1]+1] for i, start_end_inds_tuple in enumerate(list_start_end_inds_tuple)]
    nparray3d_mention_wordpiece_rep_in_sent = np.array(list_nparray2d_mention_wordpiece_reps_in_sent) # turn to 3D vectors
    #print(nparray3d_mention_wordpiece_rep_in_sent.shape,nparray3d_mention_wordpiece_rep_in_sent)
    # the contextual mention representation is the (i) average or (ii) the first subword of the wordpiece reprsentations
    # (i) averaging
    nparray2d_mention_rep_in_sent = np.mean(nparray3d_mention_wordpiece_rep_in_sent,axis=1)
    # (ii) first subword
    #nparray2d_mention_rep_in_sent = nparray3d_mention_wordpiece_rep_in_sent[:,0,:]
    print(nparray2d_mention_rep_in_sent.shape, nparray2d_mention_rep_in_sent[:,:5]) # shape (num_sent,hidden_size) and show the first five dimension for each mention rep.
    
    #3. visualisation
    #normalise the cont mention rep
    nparray2d_mention_rep_in_sent_norm = nparray2d_mention_rep_in_sent/np.linalg.norm(nparray2d_mention_rep_in_sent,axis=1)[:,None] # for [:,None], see https://stackoverflow.com/questions/37867354/in-numpy-what-does-selection-by-none-do
    print(nparray2d_mention_rep_in_sent_norm.shape, nparray2d_mention_rep_in_sent_norm[:,:5])
    #TSNE viz
    TSNE_plot_from_emb_text_adjusted(nparray2d_mention_rep_in_sent_norm,list_sent,figname2save='%s_%s_cont_emb_window%s.png' % (model_name, mention_cue, str(window_size)),fontsize=6,len_sent_anomaly_appended_trail=len_sent_anomaly_appended_trail)
    
    # BERT server shut
    shut_args = get_shutdown_parser().parse_args(['-ip','localhost',
                                                  '-port','5555',
                                                  '-timeout','5000'])
    BertServer.shutdown(shut_args)    