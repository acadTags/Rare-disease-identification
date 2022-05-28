#train and test the phenotype confirmation model - using manually labelled data or data-level fusion of weakly data and manually labelled data

#this program implements:
#(i) train with validation data and test with test data
#(ii) train with validation + weak data and test with test data
#(iii) 10-fold cross validation to tune the best number of weak data for fusion
#(iv) save the results to the sheet

# this program also export the strongly supervised models for later use in the subsequent steps of this project.

from sent_bert_emb_viz_util import load_data, encode_data_tuple, get_model_from_encoding_output, test_model_from_encoding_output
import numpy as np
import pandas as pd
from step4_further_results_from_annotations import get_and_display_results,rule_based_model_ensemble
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from sklearn.model_selection import KFold # for kfold cross validation of fully (with weak) supervised models
import os, pickle

if __name__ == '__main__':
    fill_data = True # save the supervised learning results (no weak data added)
    
    num_validation_data = 400
    trained_model_name_sup = 'model_blueBERTnorm_ws5_sup.pik' # the name of the fully/strongly supervised model to save
    
    #masked_training = False
    #use_doc_struc = True

    masking_rate = 1 #0.15 # to implement
    window_size = 5
    C = 1 # l2 regularisation parameter
    model_path='./NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12/'; model_name = 'blueBERTnorm'
    #sampling strategies - unused
    balanced_random_sampling=False
    diverse_sampling=False
    num_of_data_per_mention=25
    
    #do validation or not: to tune the best number of weak data and labelled data separately.
    do_validation_num_weak_data = False # for weak data
    weak_tr_sample_loop_step = 10 # adding this number of weak data every time
    num_kfold = 10 # k-fold cross validation on 'validation data'

    do_validation_num_sup_data = False # for labelled data (and compare the weakly trained model)
    trained_model_name_weak = 'model_blueBERTnorm_ws5_ds_nm9000m500.pik'        
    sup_tr_sample_loop_step = 10 # adding this number of labelled data every time
    # also check how the number of labelled data affect *testing* performance - this result is less convincing than validation (10fold cv) results
    do_testing_num_sup_data = False
    
    marking_str_tr = 'training'
    masking_str_tr_sup = 'training_supervised'
    marking_str_te = 'testing_1073'
    marking_str_te_final = 'testing_673' # held-out for final testing

    #trained_models_name = 'model_%s_ws%s%s%s%s.pik' % (model_name, str(window_size), '_ds' if use_doc_struc else '', '_divs%s' % str(num_of_data_per_mention) if diverse_sampling else '','_nm%sm%s' % (str(num_of_training_samples_non_masked),str(num_of_training_samples_masked)))
    
    #1. load weak data, encoding
    #load data
    data_list_tuples = load_data('mention_disamb_data.pik')
    random.Random(1234).shuffle(data_list_tuples) #randomly shuffle the list with a random seed        
    #data_list_tuples = data_list_tuples[0:num_sample] # set a small sample for quick testing
    num_weak_data_total = len(data_list_tuples)
    print(num_weak_data_total)
    #encoding of weak data
    #output_tuple_masked = encode_data_tuple(data_list_tuples, masking=True, with_doc_struc=False, model_path=model_path, marking_str=marking_str_tr, window_size=window_size, masking_rate=masking_rate, diverse_sampling=diverse_sampling, num_of_data_per_mention=num_of_data_per_mention)
    output_tuple_weak_non_masked_ds = encode_data_tuple(data_list_tuples, masking=False, with_doc_struc=True, model_path=model_path, marking_str=marking_str_tr, window_size=window_size, masking_rate=masking_rate, diverse_sampling=diverse_sampling, num_of_data_per_mention=num_of_data_per_mention)
    #get the sampled weak training data (wtr) to fuse with labelled data
    X_wtr, y_wtr, list_ind_empty_cw_wtr, list_ind_wrong_find_mt_wtr = output_tuple_weak_non_masked_ds
    #num_of_weak_tr_sample_adjusted = num_of_weak_tr_sample
    # for ind in list_ind_empty_cw_wtr + list_ind_wrong_find_mt_wtr:
        # if ind < num_of_weak_tr_sample:
            # num_of_weak_tr_sample_adjusted -= 1      
    #try only a random sample of the whole training data: the first num_of_training_samples (adjusted by the error samples), X_train is the randomly shuffled data.
    #X_wtr = X_wtr[:num_of_weak_tr_sample_adjusted,:]
    #y_wtr = y_wtr[:num_of_weak_tr_sample_adjusted]
    
    #2. encode validation (n=400) and testing (n=673) data
    df = pd.read_excel('for validation - SemEHR ori.xlsx')
    data_list_tuples_validation = []
    data_list_tuples_test = []
    for i, row in df.iterrows(): 
        doc_struc = row['document structure']
        text = row['Text']
        mention = row['mention']
        UMLS_code = row['UMLS with desc'].split()[0]
        UMLS_desc = ' '.join(row['UMLS with desc'].split()[1:])
        #label = row['manual label from Hang']
        label = row['gold text-to-UMLS label']
        label = 0 if label == -1 else label # assume that the inapplicable (-1) entries are all False.
        #print(label)
        data_tuple = (text,doc_struc,mention,UMLS_code,UMLS_desc,label)
        #if i<2:
        #    print(data_tuple)
        if i<num_validation_data:
            data_list_tuples_validation.append(data_tuple)
        else:
            data_list_tuples_test.append(data_tuple)            
    print(len(data_list_tuples_validation),len(data_list_tuples_test))
    
    #3. train and test a model
    #3.1 encode the labelled 'validation' data
    output_tuple_validation_non_masked_ds = encode_data_tuple(data_list_tuples_validation, masking=False, with_doc_struc=True, model_path=model_path, marking_str=masking_str_tr_sup, window_size=window_size, masking_rate=masking_rate)    
    output_tuple_validation_non_masked = encode_data_tuple(data_list_tuples_validation, masking=False, with_doc_struc=False, model_path=model_path, marking_str=masking_str_tr_sup, window_size=window_size, masking_rate=masking_rate)    
    
    #also split the 'validation' data
    X_mtr, y_mtr, list_ind_empty_cw_mtr, list_ind_wrong_find_mt_mtr = output_tuple_validation_non_masked_ds
        
    #carry out 10-fold cross-validation and to check the best number of weak data needed.
    if do_validation_num_weak_data:
        k_fold = KFold(num_kfold) # 10-fold cv for uspervised models (10 fold on the 'validation' data)
        plt.xlabel("labelled data + every %s weak data" % str(weak_tr_sample_loop_step))
        plt.ylabel("validation scores")
        for i, num_of_weak_tr_sample in tqdm(enumerate(range(0,num_weak_data_total,weak_tr_sample_loop_step))):
            print(num_of_weak_tr_sample)
            #train and 10-fold cross-validation on 'validation' data
            val_prec_score,val_rec_score = 0.0,0.0
            for k, (tr_ind, val_ind) in enumerate(k_fold.split(X_mtr, y_mtr)):
                #create supervised (fused with weak data) training data with k-fold split on 'validation' data
                #augment the data (manual tr, mtr) with the sampled weak data
                X_tr = np.append(X_mtr[tr_ind],X_wtr[:num_of_weak_tr_sample],axis=0)
                y_tr = np.append(y_mtr[tr_ind],y_wtr[:num_of_weak_tr_sample],axis=0)
                X_val = X_mtr[val_ind]
                y_val = y_mtr[val_ind]
                
                num_of_tr_samples = len(X_tr)
                num_of_val_samples = len(X_val)
                #train
                clf_non_masked_sup_tmp = get_model_from_encoding_output((X_tr,y_tr), num_of_tr_samples, test_size=0.0, C=1)
                #validation
                y_val_labelled_non_masked_ds, y_pred_val_labelled_non_masked_ds,_ = test_model_from_encoding_output((X_val,y_val), num_of_val_samples, clf_non_masked_sup_tmp)
                val_prec_score_,val_rec_score_,_ = get_and_display_results(y_val_labelled_non_masked_ds, y_pred_val_labelled_non_masked_ds)
                #sum results
                val_prec_score = val_prec_score + val_prec_score_
                val_rec_score = val_rec_score + val_rec_score_
            #average results
            val_prec_score = val_prec_score/num_kfold
            val_rec_score = val_rec_score/num_kfold
            val_f1_score = 2*val_prec_score*val_rec_score/(val_prec_score+val_rec_score)
            
            #plot results
            plt.scatter(num_of_weak_tr_sample/weak_tr_sample_loop_step, val_prec_score, c='red', label='precision')
            plt.scatter(num_of_weak_tr_sample/weak_tr_sample_loop_step, val_rec_score, c='green', label='recall')
            plt.scatter(num_of_weak_tr_sample/weak_tr_sample_loop_step, val_f1_score, c='blue', label='f1')
            
            if i==0:
                plt.legend()
            plt.pause(0.05)
        plt.show()
    
    #carry out 10-fold cross-validation and to check the best number of supervised data needed.
    #also compare to the the best weakly trained model
    if do_validation_num_sup_data:
        #load weakly trained model
        if os.path.exists(trained_model_name_weak):
            with open(trained_model_name_weak, 'rb') as data_f:
                clf_non_masked_ds_weak, clf_masked_weak = pickle.load(data_f)    
                
        k_fold = KFold(num_kfold) # 10-fold cv (10 fold on the 'validation' data)
        #get results with weakly trained model
        val_prec_score_w,val_rec_score_w = 0.0,0.0
        for k, (tr_ind, val_ind) in enumerate(k_fold.split(X_mtr, y_mtr)):
            X_val = X_mtr[val_ind]
            y_val = y_mtr[val_ind]
            num_of_val_samples = len(X_val)
            y_val_labelled_non_masked_ds, y_pred_val_labelled_non_masked_ds_weak,_ = test_model_from_encoding_output((X_val,y_val), num_of_val_samples, clf_non_masked_ds_weak)
            val_prec_score_w_,val_rec_score_w_,_ = get_and_display_results(y_val_labelled_non_masked_ds, y_pred_val_labelled_non_masked_ds_weak)
            #sum results
            val_prec_score_w = val_prec_score_w + val_prec_score_w_
            val_rec_score_w = val_rec_score_w + val_rec_score_w_
        #average results
        val_prec_score_w = val_prec_score_w/num_kfold
        val_rec_score_w = val_rec_score_w/num_kfold
        val_f1_score_w = 2*val_prec_score_w*val_rec_score_w/(val_prec_score_w+val_rec_score_w)
            
        #get results with fully supervised model
        plt.xlabel("number of labelled data")
        plt.ylabel("validation scores")
        for i, num_of_sup_tr_sample in tqdm(enumerate(range(sup_tr_sample_loop_step,num_validation_data,sup_tr_sample_loop_step))):
            print(num_of_sup_tr_sample)
            #train and 10-fold cross-validation on 'validation' data
            val_prec_score,val_rec_score = 0.0,0.0
            for k, (tr_ind, val_ind) in enumerate(k_fold.split(X_mtr, y_mtr)):
                #create supervised (fused with weak data) training data with k-fold split on 'validation' data
                #augment the data (manual tr, mtr) with the sampled weak data
                X_tr = X_mtr[tr_ind][:num_of_sup_tr_sample,:]
                y_tr = y_mtr[tr_ind][:num_of_sup_tr_sample]
                X_val = X_mtr[val_ind]
                y_val = y_mtr[val_ind]
                
                # check if there are more than one classes in the data - if not, continue to the next one.
                if len(np.unique(y_tr)) == 1:
                    continue
                
                num_of_tr_samples = len(X_tr)
                num_of_val_samples = len(X_val)
                #train
                clf_non_masked_sup_tmp = get_model_from_encoding_output((X_tr,y_tr), num_of_tr_samples, test_size=0.0, C=1)
                #validation
                y_val_labelled_non_masked_ds, y_pred_val_labelled_non_masked_ds,_ = test_model_from_encoding_output((X_val,y_val), num_of_val_samples, clf_non_masked_sup_tmp)
                val_prec_score_,val_rec_score_,_ = get_and_display_results(y_val_labelled_non_masked_ds, y_pred_val_labelled_non_masked_ds)
                #sum results
                val_prec_score = val_prec_score + val_prec_score_
                val_rec_score = val_rec_score + val_rec_score_
            #average results
            val_prec_score = val_prec_score/num_kfold
            val_rec_score = val_rec_score/num_kfold
            val_f1_score = 2*val_prec_score*val_rec_score/(val_prec_score+val_rec_score)
            
            #plot results (fully supervised model) as scattered dots
            x_axis_position = num_of_tr_samples
            plt.scatter(x_axis_position, val_prec_score, c='red', label='P (strong)')
            plt.scatter(x_axis_position, val_rec_score, c='green', label='R (strong)')
            plt.scatter(x_axis_position, val_f1_score, c='blue', label='F1 (strong)')
            #plot results (weakly supervised model) as horizontal line
            plt.hlines(val_prec_score_w,0,x_axis_position,'r',label='P (weak)')
            plt.hlines(val_rec_score_w,0,x_axis_position,'g',label='R (weak)')
            plt.hlines(val_f1_score_w,0,x_axis_position,'b',label='F1 (weak)')
            #plt.plot(x_axis_position, val_prec_score_w, 'r--',label='P (weak)')
            #plt.plot(x_axis_position, val_rec_score_w, 'g--',label='R (weak)')
            #plt.plot(x_axis_position, val_f1_score_w, 'b--',label='F1 (weak)')
            
            if i==0:
                plt.legend()
            plt.pause(0.05)
        plt.show()
        pass # maybe to do later
        
    #3.2 encode the final test set, train and test to see the result
    output_tuple_test_non_masked_ds = encode_data_tuple(data_list_tuples_test, masking=False, with_doc_struc=True, model_path=model_path, marking_str=marking_str_te_final, window_size=window_size, masking_rate=masking_rate)
    output_tuple_test_non_masked = encode_data_tuple(data_list_tuples_test, masking=False, with_doc_struc=False, model_path=model_path, marking_str=marking_str_te_final, window_size=window_size, masking_rate=masking_rate)
    
    #plot the number of labelled data affecting the testing performance
    if do_testing_num_sup_data:
        #plt.xlabel("adding every %s labelled data" % str(sup_tr_sample_loop_step))
        plt.xlabel("Number of strongly labelled data")
        plt.ylabel("Testing scores")
        ax = plt.gca()
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        markersize = 20
        
        #get results with weakly trained model
        #load weakly trained model
        if os.path.exists(trained_model_name_weak):
            with open(trained_model_name_weak, 'rb') as data_f:
                clf_non_masked_ds_weak, clf_masked_weak = pickle.load(data_f)    
                
        y_test_labelled_non_masked_ds, y_pred_test_labelled_non_masked_ds_weak,_ = test_model_from_encoding_output(output_tuple_test_non_masked_ds, len(data_list_tuples_test), clf_non_masked_ds_weak)
        test_prec_score_w,test_rec_score_w,test_f1_score_w = get_and_display_results(y_test_labelled_non_masked_ds, y_pred_test_labelled_non_masked_ds_weak)
        
        k_fold = KFold(num_kfold) # 10-fold cv for validation data shifting
        #get results with fully supervised model
        for i, num_of_sup_tr_sample in tqdm(enumerate(range(sup_tr_sample_loop_step,num_validation_data+sup_tr_sample_loop_step,sup_tr_sample_loop_step))):
            print('The first %s data' % num_of_sup_tr_sample)
            test_prec_score,test_rec_score = 0.0,0.0
            num_runs = num_kfold
            for k, (tr_ind, tr_ind_) in enumerate(k_fold.split(X_mtr, y_mtr)):
                tr_ind_all_shifted = np.concatenate((tr_ind, tr_ind_), axis=None)
                x_mtr_shifted = X_mtr[tr_ind_all_shifted]
                y_mtr_shifted = y_mtr[tr_ind_all_shifted]
                
                # select the first num_of_sup_tr_sample data for training
                X_mtr_selected, y_mtr_selected = x_mtr_shifted[:num_of_sup_tr_sample,:],y_mtr_shifted[:num_of_sup_tr_sample]
                num_of_actual_sup_tr_sample = X_mtr_selected.shape[0] # the actual tr sample may be less than num_of_sup_tr_sample for the last loop due to err samples during the encoding process
                
                # check if there are more than one classes in the data - if not, continue to the next one.
                if len(np.unique(y_mtr_selected)) == 1:
                    num_runs = num_runs - 1
                    continue
                    
                # train
                clf_non_masked_sup_tmp = get_model_from_encoding_output((X_mtr_selected, y_mtr_selected), num_of_actual_sup_tr_sample, test_size=0.0, C=1)
                # test
                y_test_labelled_non_masked_ds, y_pred_test_labelled_non_masked_ds_tmp,list_of_err_samples_non_masked = test_model_from_encoding_output(output_tuple_test_non_masked_ds, len(data_list_tuples_test), clf_non_masked_sup_tmp)
                test_prec_score_,test_rec_score_,test_f1_score = get_and_display_results(y_test_labelled_non_masked_ds, y_pred_test_labelled_non_masked_ds_tmp)
                #sum results
                test_prec_score = test_prec_score + test_prec_score_
                test_rec_score = test_rec_score + test_rec_score_
            #average results
            test_prec_score = test_prec_score/num_runs
            test_rec_score = test_rec_score/num_runs
            test_f1_score = 2*test_prec_score*test_rec_score/(test_prec_score+test_rec_score) if test_prec_score+test_rec_score != 0.0 else 0.0
            print('averaged results of %d fold (%d runs): prec %s rec %s f1 %s' % (num_kfold,num_runs,test_prec_score,test_rec_score,test_f1_score)) 
            
            #plot results (fully supervised model) as scattered dots
            x_axis_position = num_of_actual_sup_tr_sample
            plt.scatter(x_axis_position, test_prec_score, s=markersize, marker='x', c='red', label='P (strong)')
            plt.scatter(x_axis_position, test_rec_score, s=markersize, marker= '+', c='blue', label='R (strong)')
            plt.scatter(x_axis_position, test_f1_score, s=markersize, marker='.', c='green', label='F1 (strong)')
            #plt.plot(x_axis_position, test_prec_score, c='red', label='precision')
            #plt.plot(x_axis_position, test_rec_score, c='green', label='recall')
            #plt.plot(x_axis_position, test_f1_score, c='blue', label='f1')
            #plot results (weakly supervised model) as horizontal line
            plt.hlines(test_prec_score_w,0,x_axis_position,'red',linestyles='dotted',label='P (weak)')
            plt.hlines(test_rec_score_w,0,x_axis_position,'blue',linestyles='dashed',label='R (weak)')
            plt.hlines(test_f1_score_w,0,x_axis_position,'green',linestyles='solid',label='F1 (weak)')
            if i==0:
                plt.legend()
            plt.pause(0.05)
        plt.savefig('manual every10 - testing results - 10fold val sup.png', dpi=300) # used as the figure in the paper
        plt.show()
        
    #get the model trained with the full validation set (n=400)
    clf_non_masked_ds_sup = get_model_from_encoding_output(output_tuple_validation_non_masked_ds, len(data_list_tuples_validation), test_size=0.0, C=1)
    clf_non_masked_sup = get_model_from_encoding_output(output_tuple_validation_non_masked, len(data_list_tuples_validation), test_size=0.0, C=1)
    
    #save the strongly supervised model
    with open(trained_model_name_sup, 'wb') as data_f:
        pickle.dump((clf_non_masked_ds_sup,clf_non_masked_sup), data_f)
        print('\n' + trained_model_name_sup, 'saved')
        
    #test on the test set (n=673 - shown as 672 due to one err sample)
    print('non-masked ds results:')
    y_test_labelled_non_masked_ds, y_pred_test_labelled_non_masked_ds,list_of_err_samples_non_masked_ds = test_model_from_encoding_output(output_tuple_test_non_masked_ds, len(data_list_tuples_test), clf_non_masked_ds_sup)
    test_prec_score,test_rec_score,test_f1_score = get_and_display_results(y_test_labelled_non_masked_ds, y_pred_test_labelled_non_masked_ds)
    
    print('non-masked results:')
    y_test_labelled_non_masked, y_pred_test_labelled_non_masked,list_of_err_samples_non_masked = test_model_from_encoding_output(output_tuple_test_non_masked, len(data_list_tuples_test), clf_non_masked_sup)
    test_prec_score,test_rec_score,test_f1_score = get_and_display_results(y_test_labelled_non_masked, y_pred_test_labelled_non_masked)
    
    if fill_data:
        #fill non-masked ds results
        masked_training=False;use_doc_struc=True
        y_pred_test = y_pred_test_labelled_non_masked_ds
        print('df_length:',len(df))
        print('y_pred_test:',len(y_pred_test))
        #fill the prediction into the .xlsx file
        ind_y_pred_test=0
        result_column_name = 'full supervised model %s prediction%s%s' % (model_name, ' (masked training)' if masked_training else '', ' ds' if use_doc_struc else '')
        if not result_column_name in df.columns:
            df[result_column_name] = ""
        for i, row in df.iterrows():
            if i < num_validation_data: #only fill the testing data
                continue
            if i-num_validation_data in list_of_err_samples_non_masked_ds: # shift the index to the testing data index
                continue
            if row[result_column_name] != y_pred_test[ind_y_pred_test]:
                print('row %s results changed %s to %s' % (str(i), row[result_column_name], y_pred_test[ind_y_pred_test]))
            df.at[i,result_column_name] = y_pred_test[ind_y_pred_test]
            ind_y_pred_test = ind_y_pred_test + 1
            
        #fill non-masked results
        masked_training=False;use_doc_struc=False
        y_pred_test = y_pred_test_labelled_non_masked
        print('df_length:',len(df))
        print('y_pred_test:',len(y_pred_test))
        #fill the prediction into the .xlsx file
        ind_y_pred_test=0
        result_column_name = 'full supervised model %s prediction%s%s' % (model_name, ' (masked training)' if masked_training else '', ' ds' if use_doc_struc else '')
        if not result_column_name in df.columns:
            df[result_column_name] = ""
        for i, row in df.iterrows():
            if i < num_validation_data: #only fill the testing data
                continue
            if i-num_validation_data in list_of_err_samples_non_masked: # shift the index to the testing data index
                continue
            if row[result_column_name] != y_pred_test[ind_y_pred_test]:
                print('row %s results changed %s to %s' % (str(i), row[result_column_name], y_pred_test[ind_y_pred_test]))
            df.at[i,result_column_name] = y_pred_test[ind_y_pred_test]
            ind_y_pred_test = ind_y_pred_test + 1    
            
        df.to_excel('for validation - SemEHR ori - predicted - sup.xlsx',index=False)