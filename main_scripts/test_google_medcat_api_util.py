import requests
import json
#from sent_bert_emb_viz_util import get_char_offset

def get_json_google_healthcare_api(doc, gcloud_access_token):
    # remove single quote significant
    doc = doc.replace('\'','')
    
    headers = {
        'Authorization': 'Bearer %s' % gcloud_access_token,
        'Content-Type': 'application/json',
    }

    data = '{ \'nlpService\':\'projects/[username]/locations/[server name]/services/nlp\', \'documentContent\':\'%s\' }' % doc

    response = requests.post('https://healthcare.googleapis.com/v1beta1/projects/[username]/locations/[server name]/services/nlp:analyzeEntities', headers=headers, data=data)
    #documentation https://cloud.google.com/healthcare/docs/reference/rest/v1beta1/projects.locations.services.nlp/analyzeEntities
    
    return response#.text

# check if the json return an error message, if so return the error code with 'err', else return empty string.
def check_if_error(json_output_str):
    json_output = json.loads(json_output_str) 
    
    if json_output.get('error', None) != None:
        err = json_output['error']
        if err.get('code',None) != None:
            err_code = err['code']
        else:
            err_code = ''
    else:
        return ''
    return 'err' + str(err_code)
        

# from G-API        
def get_entities(json_output_str,mention_cue='',ment_begin_offset=0,ment_end_offset=0,tolerance=0):
    ''' get the entities of a mention cue based on the output and semantic type
    input: json_output from GOOGLE Healthcare API as string, mention_cue, ment_begin_offset and ment_end_offset in the context window
    output: (i) the dict of UMLS Ids in GOOGLE API format to their last occurrence mention-level confidence score, and,
            (ii) the mention info: ent_ment_text, ment_type, list_umls_entity, temporality, certainty, subject, confidence_ment'''
    json_output = json.loads(json_output_str)
    
    list_ent_ment = json_output['entityMentions'] if json_output.get('entityMentions', None) != None else [] # list of the entity mentions, each also includes the information about this mention, e.g. temporality, certainty, subject, etc. # if there is no mentions detected, then set it as an empty list [].
    #list_ent = json_output['entities'] # list of unique entities matched to the sequence 
    
    #get the UMLS IDs for the speficied mention cue
    dict_entId_retrieved = {} # a dictionary of entity UMLS IDs.
    dict_mention_retrieved = {} # mention information
    for ent_ment in list_ent_ment:
        ent_ment_text = ent_ment['text']['content']
        begin_offset = ent_ment['text']['beginOffset'] if ent_ment['text'].get('beginOffset', None) != None else 0 # the begin offset is the index of the starting character of this mention; not exist if 0.
        end_offset = begin_offset + len(ent_ment_text) - 1
        # filter by semantic type: PROBLEM only - to be commented out if regardless of semantic types
        #if ent_ment['type'] != 'PROBLEM':
        #    continue
            
        # filter by a user-specified mention cue, if set as empty string, then all included
        #if mention_cue.lower() in ent_ment_text.lower():
        #if ment_begin_offset>=begin_offset and ment_end_offset<=end_offset: # if the pre-defined mention cue position is within the mention position detected by the Google API - this can ignore the multiword mentions that being separately detected in Google API (26 Apr 2021) - this was used for the EMBC paper - now addressed below (using a tolerance value)
        if (ment_begin_offset>=begin_offset or abs(ment_begin_offset - begin_offset) <= tolerance) and (ment_end_offset<=end_offset or abs(ment_end_offset - end_offset) <= tolerance): # either entity within the mention or entity slightly beyond the mention within a threshold: for tolerance as 0, this won't make a difference
            #print(ent_ment)
            
            ment_type = ent_ment['type']
            # a detected mention can have no linked entities, temporality, certainty, or subject
            list_umls_entity = ent_ment['linkedEntities'] if ent_ment.get('linkedEntities', None) != None else []
            temporality = ent_ment['temporalAssessment']['value'] if ent_ment.get('temporalAssessment', None) != None else ''
            certainty = ent_ment['certaintyAssessment']['value'] if ent_ment.get('certaintyAssessment', None) != None else ''
            subject = ent_ment['subject']['value'] if ent_ment.get('subject', None) != None else ''
            confidence_ment = ent_ment['confidence'] # the mention-level confidence score
            print(mention_cue, ent_ment_text, ment_type, list_umls_entity, temporality, certainty, subject, confidence_ment)
            
            dict_mention_retrieved[(ent_ment_text, ment_type, temporality, certainty, subject, confidence_ment)] = 1
            # filter by certainty and subject
            if (certainty == 'LIKELY' or certainty == 'SOMEWHAT_LIKELY') and subject == 'PATIENT':
                for umls_entity in list_umls_entity:
                    umls_entityId = umls_entity['entityId']
                    dict_entId_retrieved[umls_entityId] = confidence_ment # this will be the confidence score of the last identified UMLS ID if it was identified multiple times in the sequence
                #print(umls_entity)
    
    #get the preferred term for the retrieved UMLS IDs            
    #for ent_retrieved in dict_entId_retrieved.keys():
    #    for ent_unique in list_ent:
    #        if ent_retrieved == ent_unique['entityId']:
    #            print(ent_retrieved, ent_unique['preferredTerm'])
    
    return dict_entId_retrieved, dict_mention_retrieved#list(dict_entId_retrieved.keys())

#from MedCAT    
def get_entities_from_MedCAT_outputs(json_output_str,mention_cue='',ment_begin_offset=0,ment_end_offset=0,tolerance=0,acc_threshold=0,use_meta_ann=True):
    ''' get the entities of a mention cue based on the output and semantic type
    input: json_output from MedCAT as string, mention_cue, ment_begin_offset and ment_end_offset in the context window, offset matching tolerance, accuracy threshold (defualt as 0, no filtering here), whether or not using metaAnnotation in MedCAT for negation filtering
    output: (i) the dict of UMLS Ids in GOOGLE API format to their last occurrence mention-level confidence score, and,
            (ii) the mention info: ent_ment_text, ment_type, list_umls_entity, temporality, certainty, subject, confidence_ment'''
    json_output = json.loads(json_output_str)
    
    dict_ent_ment = json_output['entities'] if json_output.get('entities', None) != None else [] 
    
    #get the UMLS IDs for the speficied mention cue
    dict_entId_retrieved = {} # a dictionary of entity UMLS IDs.
    dict_mention_retrieved = {} # mention information
    for ent_ment in dict_ent_ment.values():
        ent_ment_text = ent_ment['detected_name']
        begin_offset = ent_ment['start']
        end_offset = ent_ment['end']#begin_offset + len(ent_ment_text) - 1
        # filter by semantic type: PROBLEM only - to be commented out if regardless of semantic types
        #if ent_ment['type'] != 'PROBLEM':
        #    continue
            
        # filter by a user-specified mention cue, if set as empty string, then all included
        #if mention_cue.lower() in ent_ment_text.lower():
        #if ment_begin_offset>=begin_offset and ment_end_offset<=end_offset: # if the pre-defined mention cue position is within the mention position detected by the Google API - this can ignore the multiword mentions that being separately detected in Google API (26 Apr 2021) - this was used for the EMBC paper - now addressed below (using a tolerance value)
        if (ment_begin_offset>=begin_offset or abs(ment_begin_offset - begin_offset) <= tolerance) and (ment_end_offset<=end_offset or abs(ment_end_offset - end_offset) <= tolerance): # either entity within the mention or entity slightly beyond the mention within a threshold: for tolerance as 0, this won't make a difference
            #print(ent_ment)
            
            list_ment_type = ent_ment['types']
            ment_types = ';'.join(list_ment_type)
            umls_entity = ent_ment['cui']
            confidence_ment = ent_ment['acc'] # the mention-level confidence score: the acc value
            # get meta_anns status and filter by meta_anns status if chosen to (use_meta_ann as True)
            entity_status = 'unavailable'
            if use_meta_ann:
                entity_meta_ann = ent_ment['meta_anns']
                if entity_meta_ann.get('Status',None) != None:
                    entity_status = entity_meta_ann['Status']['value']
            
            print(mention_cue, ent_ment_text, list_ment_type, umls_entity, confidence_ment, 'status:' + entity_status)
            
            dict_mention_retrieved[(ent_ment_text, ment_types, confidence_ment, 'status:' + entity_status)] = 1
            
            if (not use_meta_ann) or (use_meta_ann and entity_status == 'Affirmed'):
                if confidence_ment >= acc_threshold: # acc_threshold defualt as 0, no filtering here
                    # save the highest confidence/accurarcy score if there are more than one matchings
                    if dict_entId_retrieved.get(umls_entity,None) != None:
                        if dict_entId_retrieved[umls_entity] < confidence_ment:
                            dict_entId_retrieved[umls_entity] = confidence_ment
                    else:
                        dict_entId_retrieved[umls_entity] = confidence_ment
                    #dict_entId_retrieved[umls_entity] = confidence_ment # this will be the confidence score of the last identified UMLS ID if it was identified multiple times in the sequence within the mention range - but this is unlikely    
                #print(umls_entity)
    
    #get the preferred term for the retrieved UMLS IDs            
    #for ent_retrieved in dict_entId_retrieved.keys():
    #    for ent_unique in list_ent:
    #        if ent_retrieved == ent_unique['entityId']:
    #            print(ent_retrieved, ent_unique['preferredTerm'])
    
    return dict_entId_retrieved, dict_mention_retrieved#list(dict_entId_retrieved.keys())

def get_umls_desc_MedCAT(json_output_str,list_umls_ids):
    '''get the UMLS descriptions, i.e. preferred terms, from the list of umls Ids in the MedCAT output
    input: the json string of the sequence processed by MedCAT, the list of UMLS_IDs,
    output: the list of umls descriptions, one-to-one matching to the list of UMLS_IDs'''
    json_output = json.loads(json_output_str)
    dict_ent = json_output['entities'] if json_output.get('entities', None) != None else [] # list of unique entities matched to the sequence # if there is no mentions detected, then set it as an empty list [].
    list_umls_desc = []
    for umls_id in list_umls_ids:
        for ent_unique in dict_ent.values():
            if umls_id == ent_unique['cui']:
                #print(umls_id, ent_unique['preferredTerm'])
                list_umls_desc.append(ent_unique['pretty_name'])
    return list_umls_desc
    
def get_umls_desc(json_output_str,list_umls_ids):
    '''get the UMLS descriptions, i.e. preferred terms, from the list of umls Ids in the google Healthcare API json output
    input: the json string of the sequence processed by Google healthcare API, the list of UMLS_IDs,
    output: the list of umls descriptions, one-to-one matching to the list of UMLS_IDs'''
    json_output = json.loads(json_output_str)
    list_ent = json_output['entities'] if json_output.get('entities', None) != None else [] # list of unique entities matched to the sequence # if there is no mentions detected, then set it as an empty list [].
    list_umls_desc = []
    for umls_id in list_umls_ids:
        for ent_unique in list_ent:
            if umls_id == ent_unique['entityId']:
                #print(umls_id, ent_unique['preferredTerm'])
                if 'preferredTerm' in ent_unique:
                    list_umls_desc.append(ent_unique['preferredTerm'])
                else: # if no preferredTerm exists
                    list_umls_desc.append('')
    return list_umls_desc
    
def umls_id2cui(umls_id):
    ''' from a umls ID (google API format) to CUI
        for example: UMLS/34155 --> C0034155
        return empty string if the input is not a UMLS ID of google API format'''
    if umls_id[:5] == 'UMLS/':
        id = umls_id[5:]
        num_zeros = 8 - 1 - len(id)
        zeros = '0'*num_zeros
        return 'C' + zeros + id
    else:
        return ''

#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    with open(file_name, 'w', encoding="utf-8-sig") as f_output:
        f_output.write(str)
        
if __name__ == '__main__':
    # update the access token before running this
    # simply paste the output of the command here: gcloud auth application-default print-access-token
    gcloud_access_token = '[put your access token]'

    #json_output_str = get_json_google_healthcare_api('Notable OSH labs: influenza A and B: neg, urine legionella neg, urine strep pneum antigen neg. Urine cx neg.',gcloud_access_token)
    #json_output_str = get_json_google_healthcare_api('"Discharge Diagnosis: Aortic insufficiency s/p Redo aortic valve replacement Past Medical History - Hypertension - Dyslipidemia - Fatty Liver Past Surgical History - [**2200-5-20**] Replacement of Ascending Aorta with Resuspension of Aortic Valve - [**Last Name (un) 8509**] surgery - Tooth Extractions "',gcloud_access_token)
    
    #the example in http://jekyll.inf.ed.ac.uk/edieviz
    json_output_str = get_json_google_healthcare_api('There is loss of the neuronal tissue in the left inferior frontal and superior temporal lobes, consistent with a prior infarct. There is generalised cerebral volume loss which appears within normal limits for the patients age, with no focal element to the generalised atrophy. Major intracranial vessels appear patent. White matter of the brain appears largely normal, with no evidence of significant small vessel disease. No mass lesion, hydrocephalus or extra axial collection',gcloud_access_token)
    
    print(json_output_str)
    
    json_output = json.loads(json_output_str)
    for key in json_output.keys():
        print(key)
        for sub_key in json_output[key][0].keys():
            print('\t' + sub_key)
    
    #get_entities(json_output_str,'Dyslipidemia')
    print(get_entities(json_output_str))
