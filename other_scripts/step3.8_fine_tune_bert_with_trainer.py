# using huggingface transformer to fine-tune BERT for rare disease entity linking
# a mentionBERT is created, here we use the contextual mention representation (average of token represetntations in the mention of the second last layer of BERT).

from sent_bert_emb_viz_util import load_data, get_context_window, get_context_window_from_data_list_tuples, find_sub_list
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn import functional as F

#from transformers import DistilBertTokenizerFast
#from transformers import DistilBertForSequenceClassification

from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModel#, AutoModelForSequenceClassification

import pandas as pd

#0. settings
# for training and model related
bert_model_name="bert-models/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12";per_device_train_batch_size=16;per_device_eval_batch_size=64;model_name_short = 'blueBERTnorm'
#bert_model_name="bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16";per_device_train_batch_size=4;per_device_eval_batch_size=4;model_name_short = 'blueBERTlarge'
num_train_epochs = 3
bert_frozen = False
train_model = True
checkpoint_path = 'rd-fine-tune-ckpts-and-res'
# for mention representation inside the model
use_doc_struc = False
window_size = 5
masking = False
saved_model_path = './fine-tuned-rd-ph-model%s%s' % ('-masked' if masking else '','-ds' if use_doc_struc else '') # saved best model path

fill_data=True #if True, fill the prediction into the .xlsx file

#1. load data, encoding, and train model 
#load data
data_list_tuples = load_data('mention_disamb_data.pik') #data_list_tuples, a 6-element tuple of section_text, document_structure_name, mention_name, UMLS_code, UMLS_desc, label (True or False)
random.Random(1234).shuffle(data_list_tuples) #randomly shuffle the list with a random seed        
num_sample_tr = len(data_list_tuples) #9000 #len(data_list_tuples) #500 #len(data_list_tuples) #1000
num_sample_eval = 2000

# #get context window, and insert the mention offsets in to the new data_list_tuples
# #data format updated: a 6-element tuple of context window, start_mention_offset, end_mention_offset, document_structure_name, mention_name, label (True or False)
# data_list_tuples_cw = []
# for data_tuple in data_list_tuples:
    # text_snippet_context_window, ment_begin_offset, ment_end_offset = get_context_window(data_tuple[2].lower(), data_tuple[0], window_size, masking)
    # doc_struc_name = data_tuple[1]
    # mention_name = data_tuple[2]
    # label = data_tuple[5]
    # data_list_tuples_cw.append((text_snippet_context_window, ment_begin_offset, ment_end_offset, doc_struc_name, mention_name, label))

data_list_tuples_cw = get_context_window_from_data_list_tuples(data_list_tuples,window_size=window_size,masking=masking)    
data_list_tuples_cw_train, data_list_tuples_cw_valid_in_train = train_test_split(data_list_tuples_cw, test_size=0.1, random_state=1234)
data_list_tuples_cw_train = data_list_tuples_cw_train[0:num_sample_tr] # only train on a fixed part of randomly shuffled data
data_list_tuples_cw_valid_in_train = data_list_tuples_cw_valid_in_train[0:num_sample_eval] # only eval (in train) on a fixed part of randomly shuffled data

print(len(data_list_tuples_cw_valid_in_train))

class DatasetMentFiltGen(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_list_tuples_cw, use_doc_struc, verbo=True):
        'Initialization'
        
        self.list_sent_cw = [data_tuple_cw[0] for data_tuple_cw in data_list_tuples_cw if data_tuple_cw[0] != '' ]
        #self.list_offset_start = [data_tuple_cw[1] for data_tuple_cw in data_list_tuples_cw if data_tuple_cw[0] != '' ]
        #self.list_offset_end = [data_tuple_cw[2] for data_tuple_cw in data_list_tuples_cw if data_tuple_cw[0] != '' ]
        self.list_doc_struc = [data_tuple_cw[3] for data_tuple_cw in data_list_tuples_cw if data_tuple_cw[0] != '' ]
        self.mention = [data_tuple_cw[4] for data_tuple_cw in data_list_tuples_cw if data_tuple_cw[0] != '' ]
        self.list_bi_labels = [data_tuple_cw[5] for data_tuple_cw in data_list_tuples_cw if data_tuple_cw[0] != '' ]
        
        if verbo:
            print('original', len(data_list_tuples_cw), 'after eliminating empty sents', len(self.list_sent_cw))
            print(self.list_sent_cw[0:5])
            print(self.list_bi_labels[0:5])
            print('pos:', self.list_bi_labels.count(1), 'neg:', self.list_bi_labels.count(0))
        
        #tokenisation
        #tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained("bert-models/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
        if use_doc_struc:        
            self.encodings = tokenizer(self.list_doc_struc,self.list_sent_cw, truncation=True, padding=True)
        else:
            self.encodings = tokenizer(self.list_sent_cw, truncation=True, padding=True)
        #print('self.encodings:',self.encodings.items())
        self.labels = [int(bool_label) for bool_label in self.list_bi_labels]
        
        #get the new mention offsets after the tokenisation
        mention_encodings = tokenizer(self.mention, add_special_tokens=False, truncation=True, padding=False)
        #print(mention_encodings)
        #to do 
        
        token_input_ids_cw = self.encodings['input_ids']
        token_input_ids_men = mention_encodings['input_ids']
        #print('token_input_ids_cw',token_input_ids_cw)
        self.list_offset_tuples = [find_sub_list(list_men_token_id,list_cw_token_id) for list_men_token_id,list_cw_token_id in zip(token_input_ids_men,token_input_ids_cw)]
        #print(self.list_offset_tuples)
        
        self.list_offset_tuples = [offset_tuple if offset_tuple != None else (0,0) for offset_tuple in self.list_offset_tuples] # if mention tokens are not found in the context window tokens, set mention token offset as the first token, i.e. (0,0)        
        # #or remove the ones with no mention tokens found (i.e. None in self.list_offset_tuples), then do the tokenisation again get the _final encoding and offsets. - this can eliminates those in the testing data, so not recommended
        # self.list_sent_cw_final = [self.list_sent_cw[ind] for ind, offset_tuple in enumerate(self.list_offset_tuples) if offset_tuple != None]
        # self.list_doc_struc_final = [self.list_doc_struc[ind] for ind, offset_tuple in enumerate(self.list_offset_tuples) if offset_tuple != None]
        # self.mention_final = [self.mention[ind] for ind, offset_tuple in enumerate(self.list_offset_tuples) if offset_tuple != None]
        # self.labels_final = [self.labels[ind] for ind, offset_tuple in enumerate(self.list_offset_tuples) if offset_tuple != None]
        # if use_doc_struc:        
            # self.encodings_final = tokenizer(self.list_doc_struc_final,self.list_sent_cw_final, truncation=True, padding=True)
        # else:
            # self.encodings_final = tokenizer(self.list_sent_cw_final, truncation=True, padding=True)
        # #get the new mention offsets after the tokenisation
        # mention_encodings_final = tokenizer(self.mention_final, add_special_tokens=False, truncation=True, padding=False)        
        # token_input_ids_cw_final = self.encodings_final['input_ids']
        # token_input_ids_men_final = mention_encodings_final['input_ids']
        # #print(token_input_ids_cw)
        # self.list_offset_tuples_final = [find_sub_list(list_men_token_id,list_cw_token_id) for list_men_token_id,list_cw_token_id in zip(token_input_ids_men_final,token_input_ids_cw_final)]
        # print(self.list_offset_tuples_final)
        # print('%d out of %d retained after removing non-matched mention tokens' % (len(self.list_offset_tuples_final),len(self.list_offset_tuples)))
        
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_sent_cw)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label        
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        item['begin_offsets'] = self.list_offset_tuples[index][0]
        item['end_offsets'] = self.list_offset_tuples[index][1]
        return item

train_dataset = DatasetMentFiltGen(data_list_tuples_cw_train, use_doc_struc)
val_for_tr_dataset = DatasetMentFiltGen(data_list_tuples_cw_valid_in_train, use_doc_struc)

#model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
#model = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
#model = AutoModelForSequenceClassification.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")

class mentionBERT(nn.Module):
    def __init__(self):
          super(mentionBERT, self).__init__()
          self.bert = AutoModel.from_pretrained(bert_model_name)
          
          self.dense = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
          self.activation = nn.Tanh()
          
          self.classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.num_labels)
          #self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
          
    def forward(self, input_ids=None,
        attention_mask=None,
        begin_offsets=0,
        end_offsets=0, # add begin and ending offsets as input here
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True, # here as True
        return_dict=None,
    ):
        outputs = self.bert(
           input_ids, 
           attention_mask=attention_mask,output_hidden_states=output_hidden_states)
        
        hidden_states = outputs.hidden_states # a tuple of k+1 layers (k=12 for bert-base) and each is a shape of (batch_size,sent_token_len,hidden_note_size)
        second_to_last_layer_hs = hidden_states[-2]
        #print('second_to_last_layer_hs',second_to_last_layer_hs)

        #logits = self.classifier(second_to_last_layer_hs[:,0,:].view(-1,768)) ## extract the 1st token's embeddings
        
        #print('begin_offsets, end_offsets', begin_offsets, end_offsets)
        sequence_output_cont_emb = [torch.mean(second_to_last_layer_hs[ind][offset_start:offset_end+1],dim=0) for ind, (offset_start, offset_end) in enumerate(zip(begin_offsets, end_offsets))] # here offset_end needs to add 1, since the offsets from find_sub_list() subtracted 1 for the end offset
        #print('sequence_output_cont_emb',sequence_output_cont_emb)
        sequence_output_cont_emb = torch.stack(sequence_output_cont_emb)
        
        # here also has a dense layer with tanh activation - HD
        pooled_output = self.dense(sequence_output_cont_emb) 
        pooled_output = self.activation(pooled_output)
        
        logits = self.classifier(pooled_output)
        #logits = self.classifier(sequence_output_cont_emb)
        
        loss_fct = CrossEntropyLoss()
        #loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.bert.config.num_labels), labels.view(-1))
        #loss = loss_fct(logits.view(-1), labels.view(-1).float())
        
        output = (logits,) + outputs[2:]
        return ((loss,) + output)
       
model = mentionBERT()
        
#if keep the model part frozen - see https://huggingface.co/transformers/training.html
if bert_frozen:
    for param in model.bert.base_model.parameters():
        param.requires_grad = False

def compute_metrics(pred):
    labels = pred.label_ids
    #print('pred.predictions',pred.predictions)
    
    preds = pred.predictions[0].argmax(-1) # get the binary prediction from the softmax output # here get the first element of predictions as the hidden_states are also predicted
    #preds = (pred.predictions[0] > 0).astype(int) # for BCEWithLogitsLoss
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    #acc = accuracy_score(labels, preds)
    confusion_mat_tuple = confusion_matrix(labels, preds).ravel()
    #only unpack the confusion matrix when there are enough to unpack
    if len(confusion_mat_tuple) == 4:
        tn, fp, fn, tp = confusion_mat_tuple
    else:
        tn, fp, fn, tp = None, None, None, None
    return {
        #'accuracy': acc,
        'tn': tn, 
        'fp': fp, 
        'fn': fn, 
        'tp': tp,
        'precision': precision,
        'recall': recall,        
        'f1': f1
    }

# example code from https://huggingface.co/transformers/main_classes/trainer.html#trainer for multi-label classification
# class MultilabelTrainer(Trainer):
    # def compute_loss(self, model, inputs, return_outputs=False):
        # print('self.model.config.num_labels:',self.model.config.num_labels)
        # labels = inputs.pop("labels")
        # outputs = model(**inputs)
        # print('outputs:',outputs)
        # logits = outputs.logits
        # print('logits:',logits.shape)
        # loss_fct = nn.BCEWithLogitsLoss()
        # print('logits.view(-1, self.model.config.num_labels):',logits.view(-1, self.model.config.num_labels).shape)
        # print('labels.float().view(-1, self.model.config.num_labels):',labels.float().view(-1, self.model.config.num_labels).shape)
        # '''logits.view(-1, self.model.config.num_labels): torch.Size([16, 2])
           # labels.float().view(-1, self.model.config.num_labels): torch.Size([8, 2])''' # this is due to the format of single label, which is not suitable to multilabel
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        # labels.float().view(-1, self.model.config.num_labels))
        # print('loss:',loss.shape)                
        # return (loss, outputs) if return_outputs else loss

if train_model:
    #using with the parameters in https://huggingface.co/transformers/custom_datasets.html#fine-tuning-with-trainer
    #for a full list of the arguments, see https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    training_args = TrainingArguments(
        output_dir=checkpoint_path, #'./results',          # output directory
        overwrite_output_dir=True,       # If True, overwrite the content of the output directory.
        num_train_epochs=num_train_epochs,              # total number of training epochs
        per_device_train_batch_size=per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=per_device_eval_batch_size,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=500,                # logging step
        evaluation_strategy='steps',     # eval by step
        eval_accumulation_steps=1,      # accumulate results to CPU every k steps to save memoty
        save_strategy='epoch',           # save model every epoch
        load_best_model_at_end=True,     # load the best model at end
        metric_for_best_model='f1'       # with metric F1
    )

    trainer = Trainer(
        model=model,                         # the instantiated ðŸ¤— Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_for_tr_dataset,     # evaluation dataset
        compute_metrics=compute_metrics,
    )
    trainer.train()
    #trainer.train(checkpoint_path + '/checkpoint-3792') #train from a certain checkpoint
    trainer.save_model(saved_model_path)
else:
    training_args = TrainingArguments(
        output_dir=checkpoint_path, #'./results',          # output directory    
        num_train_epochs=0.00001,        # total number of training epochs # set this to an extremely small number
        per_device_train_batch_size=per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=per_device_eval_batch_size,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=500,                # logging step
        evaluation_strategy='steps',     # eval by step
        eval_accumulation_steps=1,       # accumulate results to CPU every k steps to save memoty
        save_strategy='epoch',           # save model every epoch
        load_best_model_at_end=True,     # load the best model at end
        metric_for_best_model='f1'       # with metric F1
    )
    trainer = Trainer(
        model=model,                 # the instantiated ðŸ¤— Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_for_tr_dataset,     # evaluation dataset
        compute_metrics=compute_metrics,
    )
    #trainer.train(checkpoint_path + '/checkpoint-3792') #eval with a certain checkpoint
    trainer.train(saved_model_path)
    
print(trainer.evaluate()) # see https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.predict

#2. load testing data and predict results: 
#load data from .xlsx and save the results to a specific column
# get a list of data tuples from an annotated .xlsx file
# data format: a 6-element tuple of section_text, document_structure_name, mention_name, UMLS_code, UMLS_desc, label (True or False)
df = pd.read_excel('for validation - SemEHR ori.xlsx')
# change nan values into empty strings in the two rule-based label columns
df[['neg label: only when both rule 0','pos label: both rules applied']] = df[['neg label: only when both rule 0','pos label: both rules applied']].fillna('') 
# if the data is not labelled, i.e. nan, label it as -1 (not positive or negative)
df[['manual label from ann1']] = df[['manual label from ann1']].fillna(-1) 

data_list_tuples_valid = []
data_list_tuples_test = []
for i, row in df.iterrows():
    doc_struc = row['document structure']
    text = row['Text']
    mention = row['mention']
    UMLS_code = row['UMLS with desc'].split()[0]
    UMLS_desc = ' '.join(row['UMLS with desc'].split()[1:])
    label = row['gold text-to-UMLS label']
    label = 0 if label == -1 else label # assume that the inapplicable (-1) entries are all False.
    data_tuple = (text,doc_struc,mention,UMLS_code,UMLS_desc,label)
    if i<400:
        data_list_tuples_valid.append(data_tuple)
    else:
        data_list_tuples_test.append(data_tuple)
data_list_tuples_test_whole = data_list_tuples_valid + data_list_tuples_test
print('valid data %s, test data %d, whole eval data %s' % (len(data_list_tuples_valid),len(data_list_tuples_test),len(data_list_tuples_test_whole)))

data_list_tuples_cw_valid = get_context_window_from_data_list_tuples(data_list_tuples_valid,window_size=window_size,masking=masking)  
data_list_tuples_cw_test = get_context_window_from_data_list_tuples(data_list_tuples_test,window_size=window_size,masking=masking)  
data_list_tuples_cw_test_whole = get_context_window_from_data_list_tuples(data_list_tuples_test_whole,window_size=window_size,masking=masking)  

valid_dataset = DatasetMentFiltGen(data_list_tuples_cw_valid, use_doc_struc)
test_dataset = DatasetMentFiltGen(data_list_tuples_cw_test, use_doc_struc)
test_whole_dataset = DatasetMentFiltGen(data_list_tuples_cw_test_whole, use_doc_struc)

print(trainer.evaluate(valid_dataset))
print(trainer.evaluate(test_dataset))
print(trainer.evaluate(test_whole_dataset))

predictions,_,metrics = trainer.predict(test_whole_dataset) # see https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.predict
#print(predictions,metrics)

#fill results to the excel sheet - to do (to record results of bluebert-base-fine-tune for non-mask and mask).
list_ind_empty_cw_whole_test = [ind for ind, datalist_tuple_cw_ in enumerate(data_list_tuples_cw_test_whole) if datalist_tuple_cw_[0] == '']
y_pred_test = predictions[0].argmax(-1)
# for ind, data_list_tuple in enumerate(data_list_tuples_cw_test_whole): # to do to make this by batch
    # if data_list_tuple[0] != '':
        # test_dataset_from_one_tuple = DatasetMentFiltGen([data_list_tuple], use_doc_struc, verbo=False) # not verbo here
        # pred_,_,_ = trainer.predict(test_dataset_from_one_tuple)
        # pred = pred_[0].argmax(-1)[0]
    # else:
        # pred = 0
    # y_pred_test.append(pred)
print('y_pred_test', y_pred_test)

if fill_data:
    #fill the prediction into the .xlsx file
    result_column_name = 'model %s%s hf prediction%s%s%s' % (model_name_short, 'finetune' if bert_frozen else '', ' (masked training)' if masking else '', ' ds' if use_doc_struc else '', ' tr%s' % str(num_sample_tr) if num_sample_tr<len(data_list_tuples) else '')
    if not result_column_name in df.columns:
        df[result_column_name] = ""
    ind_y_pred_test=0
    for i, row in df.iterrows():
        if i in list_ind_empty_cw_whole_test:
            continue
        if row[result_column_name] != y_pred_test[ind_y_pred_test]:
            print('row %s results changed %s to %s' % (str(i), row[result_column_name], y_pred_test[ind_y_pred_test]))
        df.at[i,result_column_name] = y_pred_test[ind_y_pred_test]
        ind_y_pred_test=ind_y_pred_test+1
    df.to_excel('for validation - SemEHR ori - hf - predicted%s%s.xlsx' % (' - masked' if masking else '', ' - ds' if use_doc_struc else ''),index=False)
    #hf stands for huggingface, it actually means fine-tuning.