# get rare diseases from free-texts using Gate-BioYODIE and ontologies

import json
import sys
import urllib

# pip install requests
# https://pypi.org/project/requests/
import requests

from collections import defaultdict
import pandas as pd
import os
import re

import requests

# extracting umls from free text with Bio-YODIE
def umls_from_free_text(text):
    '''
    input a piece of raw text in a clinical note
    output the umls code of the current diseases or symptoms identified by BioYODIE
    the output format is a dictionary of tuples: (umls_code, in_text_mention, pref_label)
    # REFERENCES
#
# GATE Cloud on-line API
#   https://cloud.gate.ac.uk/info/help/online-api.html
#
# BioYODIE Named Entity Disambiguation
#   https://cloud.gate.ac.uk/shopfront/displayItem/bio-yodie
'''
    # display text for debugging
    #print(text)
    
    dict_umls_intext_pref_tuple = defaultdict(int)
    
    # The base URL for all GATE Cloud API services
    prefix = "https://cloud-api.gate.ac.uk/"
    service = "process-document/bio-yodie?annotations=Bio:Disease"
    url = urllib.parse.urljoin(prefix, service)
    
    headers = {'Content-Type': 'text/plain'}

    # Make the API request and raise error if status >= 4xx
    response = requests.post(url, data=text, headers=headers)
    response.raise_for_status()

    # The requests library has a method for returning parsed JSON
    gate_json = response.json()
    
    # Pretty print the response, mostly for debugging
    #print(json.dumps(gate_json, indent=2)) #If indent is a non-negative integer or string, then JSON array elements and object members will be pretty-printed with that indent level.
    
    # Find each annotation and print its type and the text it is annotating, along with the UMLS code, the captured mention and preferred label.
    print('The diseases or syndromes identified:')
    response_text = gate_json["text"]
    for annotation_type, annotations in gate_json['entities'].items():
        for annotation in annotations:
            i, j = annotation["indices"]
            if annotation["Experiencer"] == 'Patient' and annotation["Negation"] == 'Affirmed' and annotation["STY"] == "Disease or Syndrome":
                in_text_ann = response_text[i:j]
                umls_code = annotation["inst"]
                #umls_label = annotation["PREF"] # using the label from BioYODIE, but this is not accurate.
                umls_label = umls2prefLabel(umls_code) # using the SPARQL from linked open data to get the label.
                if umls_label == '':
                    umls_label = annotation["PREF"]
                
                dict_umls_intext_pref_tuple[(umls_code,in_text_ann,umls_label)] += 1
                print('\t', in_text_ann, ',', umls_code, '(%s)' % umls_label)
            
    return dict_umls_intext_pref_tuple

def display_output_json(gate_json):
    # Pretty print the response, mostly for debugging
    print(json.dumps(gate_json, indent=2)) #If indent is a non-negative integer or string, then JSON array elements and object members will be pretty-printed with that indent level.
    
    # Find each annotation and print its type and the text it is annotating
    response_text = gate_json["text"]
    for annotation_type, annotations in gate_json['entities'].items():
        for annotation in annotations:
            i, j = annotation["indices"]
            print(annotation_type, ":", response_text[i:j])

def umls2prefLabelwithDict(UMLS_code,dict_umls_preflabel=None):
    ''' use a dictionary to store and retrieve preflabel if available, otherwise call umls2prefLabel(UMLS_code) to get the preflabel.
    '''
    if dict_umls_preflabel == None: # if not specified, create an empty new dictionary
        dict_umls_preflabel = {}
    if dict_umls_preflabel.get(UMLS_code,None) == None:
        pref_label = umls2prefLabel(UMLS_code)
        dict_umls_preflabel[UMLS_code] = pref_label
    else:
        pref_label = dict_umls_preflabel[UMLS_code]
    return prefLabel, dict_umls_preflabel
    
def umls2prefLabel(UMLS_code):
    ''' example SPARQL query:
    SELECT DISTINCT ?concept
    WHERE {
    <http://linkedlifedata.com/resource/umls/id/C0020538> skos:prefLabel ?concept.
    }
    example SPARQL url:http://linkedlifedata.com/sparql.json?query=SELECT+DISTINCT+%3Fconcept%0D%0AWHERE+%7B%0D%0A%3Chttp%3A%2F%2Flinkedlifedata.com%2Fresource%2Fumls%2Fid%2FC0020538%3E+skos%3AprefLabel+%3Fconcept.%0D%0A%7D&_implicit=false&implicit=true&_form=%2Fsparql
    '''
    prefLabel=''
    #construct SPARQL query
    SPARQL_url = "http://linkedlifedata.com/sparql.json?query=SELECT+DISTINCT+%3Fconcept%0D%0AWHERE+%7B%0D%0A%3Chttp%3A%2F%2Flinkedlifedata.com%2Fresource%2Fumls%2Fid%2F" + UMLS_code + "%3E+skos%3AprefLabel+%3Fconcept.%0D%0A%7D&_implicit=false&implicit=true&_form=%2Fsparql"
    response = requests.get(SPARQL_url)
    if response.status_code == 200:
        entity_data = response.json()
        for entry in entity_data["results"]["bindings"]:
            prefLabel = entry["concept"]["value"]            
    else:
        print(UMLS_code, 'querying failed:', response.status_code)
    return prefLabel

# retrieve ICD9 based on UMLS by querying the SPARQL endpoint http://linkedlifedata.com/sparql
def umls2icd9(UMLS_code):
    '''    
    input umls code
    output tuple (icd9 code, description)
    
    #example SPARQL query:
    SELECT DISTINCT ?concept ?code ?description
    WHERE {
        <http://linkedlifedata.com/resource/umls/id/C0020538> skos-xl:altLabel ?concept.
        ?concept skos:note "ICD-9-CM".
        ?concept skos:notation ?code.
        ?concept skos-xl:literalForm ?description
    }

    #example SPARQL url http://linkedlifedata.com/sparql.json?query=PREFIX+skos%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2004%2F02%2Fskos%2Fcore%23%3E%0D%0APREFIX+rdfs%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F01%2Frdf-schema%23%3E%0D%0APREFIX+lld%3A+%3Chttp%3A%2F%2Flinkedlifedata.com%2Fresource%2F%3E%0D%0ASELECT+DISTINCT+%3Fconcept+%3Fcode+%3Fdescription%0D%0AWHERE+%7B%0D%0A%3Chttp%3A%2F%2Flinkedlifedata.com%2Fresource%2Fumls%2Fid%2FC0020538%3E+skos-xl%3AaltLabel+%3Fconcept.%0D%0A%3Fconcept+skos%3Anote+%22ICD-9-CM%22.%0D%0A%3Fconcept+skos%3Anotation+%3Fcode.%0D%0A%3Fconcept+skos-xl%3AliteralForm+%3Fdescription%0D%0A%7D%0D%0A&_implicit=false&implicit=true&_form=%2Fsparql

    '''
    dict_code_desc_tuple = {}
    #construct SPARQL query
    SPARQL_url = "http://linkedlifedata.com/sparql.json?query=PREFIX+skos%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2004%2F02%2Fskos%2Fcore%23%3E%0D%0APREFIX+rdfs%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F01%2Frdf-schema%23%3E%0D%0APREFIX+lld%3A+%3Chttp%3A%2F%2Flinkedlifedata.com%2Fresource%2F%3E%0D%0ASELECT+DISTINCT+%3Fconcept+%3Fcode+%3Fdescription%0D%0AWHERE+%7B%0D%0A%3Chttp%3A%2F%2Flinkedlifedata.com%2Fresource%2Fumls%2Fid%2F" + UMLS_code + "%3E+skos-xl%3AaltLabel+%3Fconcept.%0D%0A%3Fconcept+skos%3Anote+%22ICD-9-CM%22.%0D%0A%3Fconcept+skos%3Anotation+%3Fcode.%0D%0A%3Fconcept+skos-xl%3AliteralForm+%3Fdescription%0D%0A%7D%0D%0A&_implicit=false&implicit=true&_form=%2Fsparql"
    response = requests.get(SPARQL_url)
    if response.status_code == 200:
        entity_data = response.json()
        for entry in entity_data["results"]["bindings"]:
            code = entry["code"]["value"]
            desc = entry["description"]["value"]
            if dict_code_desc_tuple.get((code,desc), None) != None:
                dict_code_desc_tuple[(code,desc)]=dict_code_desc_tuple[(code,desc)]+1
            else:
                dict_code_desc_tuple[(code,desc)]=1
    else:
        print(UMLS_code, 'querying failed:', response.status_code)
    return dict_code_desc_tuple

#retrieve icd9 from UMLS by using bioportal https://bioportal.bioontology.org/ontologies/ICD9CM
#also output the map
def umls2icd9List_bp(UMLS_code,default_path='',map=None):
    if map is None:
        map = pd.read_csv(os.path.join(default_path,'ICD9CM_2020AB.csv'))
    matched_df=map[map['CUI'].str.contains(UMLS_code, na=False)] # will this match to multiple rows? - yes it can, and then the output will be a df of two rows
    #print(matched_df)
    #icd9_tmp = matched_df['Class ID'].to_string(index=False)
    #icd9_pref_label_tmp = matched_df['Preferred Label'].to_string(index=False)
    list_icd9,list_icd9_pref_label = [],[]
    for i,row in matched_df.iterrows():
        icd9_code_path = row['Class ID']
        icd9_code = icd9_code_path.split('/')[-1] # get the last part of the path
        list_icd9.append(icd9_code)
        list_icd9_pref_label.append(row['Preferred Label'])
    return list_icd9,list_icd9_pref_label,map
    
# retrieve ORDO based on UMLS by querying the SPARQL endpoint https://www.orpha.net/sparql
def umls2ordo(UMLS_code):
    '''input umls code
       output a tuple of (ordoID,ordoLabel)
    '''   
    '''sparql query:
        prefix obolib:<http://purl.obolibrary.org/obo/>

    select distinct ?ordoID ?relation ?ordoLabel

    where {
    {?tmp owl:annotatedSource ?ordoID;
          owl:annotatedTarget "UMLS:C0035828"^^<http://www.w3.org/2001/XMLSchema#string>}
    union
    {?tmp owl:annotatedSource ?ordoID;
          owl:annotatedTarget "UMLS:C0035828"@en}
    ?tmp  obolib:ECO_0000218 ?relation.
    ?ordoID rdfs:label ?ordoLabel.
    filter (lang(?ordoLabel) = 'en')
    } 

    LIMIT 1000
    '''
    '''sparql url https://www.orpha.net/sparql?default-graph-uri=&query=prefix+obolib%3A%3Chttp%3A%2F%2Fpurl.obolibrary.org%2Fobo%2F%3E%0D%0A%0D%0Aselect+distinct+%3FordoID+%3Frelation+%3FordoLabel%0D%0A%0D%0Awhere+%7B%0D%0A%7B%3Ftmp+owl%3AannotatedSource+%3FordoID%3B%0D%0A++++++owl%3AannotatedTarget+%22UMLS%3AC0035828%22%5E%5E%3Chttp%3A%2F%2Fwww.w3.org%2F2001%2FXMLSchema%23string%3E%7D%0D%0Aunion%0D%0A%7B%3Ftmp+owl%3AannotatedSource+%3FordoID%3B%0D%0A++++++owl%3AannotatedTarget+%22UMLS%3AC0035828%22%40en%7D%0D%0A%3Ftmp++obolib%3AECO_0000218+%3Frelation.%0D%0A%3FordoID+rdfs%3Alabel+%3FordoLabel.%0D%0Afilter+%28lang%28%3FordoLabel%29+%3D+%27en%27%29%0D%0A%7D+%0D%0A%0D%0ALIMIT+1000&should-sponge=&format=application%2Fsparql-results%2Bjson&timeout=0&debug=on'''
    dict_ordoID_label_rel_tuple = {}
    #construct SPARQL query
    SPARQL_url = "https://www.orpha.net/sparql?default-graph-uri=&query=prefix+obolib%3A%3Chttp%3A%2F%2Fpurl.obolibrary.org%2Fobo%2F%3E%0D%0A%0D%0Aselect+distinct+%3FordoID+%3Frelation+%3FordoLabel%0D%0A%0D%0Awhere+%7B%0D%0A%7B%3Ftmp+owl%3AannotatedSource+%3FordoID%3B%0D%0A++++++owl%3AannotatedTarget+%22UMLS%3A" + UMLS_code + "%22%5E%5E%3Chttp%3A%2F%2Fwww.w3.org%2F2001%2FXMLSchema%23string%3E%7D%0D%0Aunion%0D%0A%7B%3Ftmp+owl%3AannotatedSource+%3FordoID%3B%0D%0A++++++owl%3AannotatedTarget+%22UMLS%3A" + UMLS_code + "%22%40en%7D%0D%0A%3Ftmp++obolib%3AECO_0000218+%3Frelation.%0D%0A%3FordoID+rdfs%3Alabel+%3FordoLabel.%0D%0Afilter+%28lang%28%3FordoLabel%29+%3D+%27en%27%29%0D%0A%7D+%0D%0A%0D%0ALIMIT+1000&should-sponge=&format=application%2Fsparql-results%2Bjson&timeout=0&debug=on"
    response = requests.get(SPARQL_url)
    if response.status_code == 200:
        entity_data = response.json()
        for entry in entity_data["results"]["bindings"]:
            ordoID = entry["ordoID"]["value"]
            ordoLabel = entry["ordoLabel"]["value"]
            relation = entry["relation"]["value"]
            if dict_ordoID_label_rel_tuple.get((ordoID,ordoLabel,relation), None) != None:
                dict_ordoID_label_rel_tuple[(ordoID,ordoLabel,relation)]=dict_ordoID_label_rel_tuple[(ordoID,ordoLabel,relation)]+1
            else:
                dict_ordoID_label_rel_tuple[(ordoID,ordoLabel,relation)]=1
    else:
        print(UMLS_code, 'querying failed:', response.status_code)
    return dict_ordoID_label_rel_tuple

#input an ORDO_ID_url as Orphanet_3325, and choose whether using exact or narrower matching (instead of using exact+narrower+broader matching)
#output the corresponding ICD-10 codes as a list
def ordo2icd10FromJSON(ORDO_ID_url,exact_or_narrower_only=True):
    ORDO_url_for_JSON = "https://www.ebi.ac.uk/ols/api/ontologies/ordo/terms?iri=%s" % ORDO_ID_url
    response = requests.get(ORDO_url_for_JSON)
    #print(response.status_code)    
    dict_ICD10_tmp={}
    if response.status_code == 200:
        entity_data = response.json()
        #print('entity_data',entity_data)
        cross_refs = entity_data['_embedded']['terms'][0]['obo_xref']
        if cross_refs == None:
            return []
        for linked_code_entry in cross_refs:
            if linked_code_entry['database'] == 'ICD-10':
                if exact_or_narrower_only:
                    if linked_code_entry['description'][:4] == 'BTNT' or linked_code_entry['description'][:1] == 'E':
                        dict_ICD10_tmp[linked_code_entry['id']]=1
                else:
                    dict_ICD10_tmp[linked_code_entry['id']]=1
    else:
        print(ORDO_ID_url, 'querying failed:', response.status_code)
    return list(dict_ICD10_tmp.keys())
    
# output all matched ORDO ID from a UMLS code 
# from ORDO2UMLS_ICD10_ICD9+titles.csv, where the relevant columns were created from the ordo.csv file, the ORDO ontology in CSV format
# for multiple matchings, the output will be seperated by '\n' in each of the ordo_ID_tmp and ordo_pref_label_tmp
def umls2ordoFromCSV(UMLS_code,default_path='',map=None):
    #map = pd.read_excel(os.path.join(default_path,'./ontology/ORDO2UMLS_ICD10_ICD9+titles_final_v3.xlsx'),sheet_name="full sheet")
    if map is None:
        map = pd.read_excel(os.path.join(default_path,'./ontology/ORDO2UMLS_ICD10_ICD9+titles_final_v3.xlsx'),engine='openpyxl') # new coding matching map
    matched_df=map[map['UMLS IDs'].str.contains(UMLS_code)] # will this match to multiple rows? - yes it can, and then the output will be a df of two rows
    #print(matched_df)
    ordo_ID_tmp = matched_df['ORDO ID'].to_string(index=False)
    ordo_pref_label_tmp = matched_df['Preferred Label'].to_string(index=False)
    return ordo_ID_tmp, ordo_pref_label_tmp, map

# the output will be a list of ORDO IDs, instead of a string seperated by '\n'
# this function is based on def umls2ordoFromCSV()
def umls2ordoListFromCSV(UMLS_code,default_path='',map=None):
    ordo_ID_tmp, ordo_pref_label_tmp, map = umls2ordoFromCSV(UMLS_code, default_path=default_path, map=map)
    list_ordo_ID = [ordo_ID.strip() for ordo_ID in ordo_ID_tmp.split('\n')]
    list_ordo_pref_label = [ordo_pref_label.strip() for ordo_pref_label in ordo_pref_label_tmp.split('\n')]
    return list_ordo_ID, list_ordo_pref_label, map
    
#input format: (i) UMLS CUI
#              (ii) Also input the path whether the code matching sheet is stored
#              (iii) map initialised as None to store the coding matching map
#              (iv) the onto matching source, NZ (default, umls-ordo-*icd10-icd9* with MoH NZ) or bp (ORDO-*UMLS-icd9* with bioportal ICD-9-CM), or 'both' (both of the sources)    
# output all matched ICD9 from a rare disease UMLS code
# onto matching sources 
    # NZ: UMLS - ORDO - ICD 10 - ICD 9
    # bp: UMLS - ICD 9
    # both: NZ and bp

# output all matched UMLS from an ORDO ID 
# from ORDO2UMLS_ICD10_ICD9+titles.csv, where the relevant columns were created from the ordo.csv file, the ORDO ontology in CSV format
# for multiple matchings, the output will be seperated by '\n' in each of the ordo_ID_tmp and ordo_pref_label_tmp
def ordo2umlsFromCSV(ORDO_ID,default_path='',map=None):
    #map = pd.read_excel(os.path.join(default_path,'./ontology/ORDO2UMLS_ICD10_ICD9+titles.xlsx'),sheet_name="full sheet")
    if map is None:
        map = pd.read_excel(os.path.join(default_path,'./ontology/ORDO2UMLS_ICD10_ICD9+titles_final_v3.xlsx'),engine='openpyxl') # new coding matching map
    matched_df=map[map['ORDO ID'].str.endswith(ORDO_ID)] # will this match to multiple rows? - yes it could, and then the output will be a df of two rows. but for ORDO ID matching, each ORDO ID has only one row.
    #print(matched_df)
    umls_ID_tmp = matched_df['UMLS IDs'].to_string(index=False)
    return umls_ID_tmp, map
    
# from the preprocessed results in ORDO2UMLS_ICD10_ICD9+titles_final_v3.csv
def umls2ICD9FromCSV(UMLS_code,default_path='',map=None,onto_source='NZ',exact_or_narrower_only=False):  
    if map is None:
        map = pd.read_excel(os.path.join(default_path,'./ontology/ORDO2UMLS_ICD10_ICD9+titles_final_v3.xlsx'),engine='openpyxl')
    #map = pd.read_excel(os.path.join(default_path,'./ontology/ORDO2UMLS_ICD10_ICD9+titles.xlsx'),sheet_name="full sheet")
    matched_df=map[map['UMLS IDs'].str.contains(UMLS_code)] # will this match to multiple rows? - yes
    # set onto_source to 'NZ' as default if not recognisable (not NZ or bp)   
    onto_source = 'NZ' if onto_source != 'NZ' and onto_source != 'bp' and onto_source != 'both' else onto_source  
    if exact_or_narrower_only and onto_source == 'NZ': 
        # update the source of NZ to exact or narrower only
        onto_source = 'NZ-E-N'
    if onto_source == 'NZ':
        icd9_tmp = matched_df['icd9-NZ'].to_string(index=False)
        icd9_long_tit_tmp = matched_df['icd9-long-titles'].to_string(index=False)
    elif onto_source == 'NZ-E-N':
        icd9_tmp = matched_df['icd9-NZ-E-N'].to_string(index=False)
        icd9_long_tit_tmp = matched_df['icd9-long-titles-E-N'].to_string(index=False)
    elif onto_source == 'bp':
        icd9_tmp = matched_df['icd9-bp'].to_string(index=False)
        icd9_long_tit_tmp = matched_df['icd9-pref-label-bp'].to_string(index=False)
    elif onto_source == 'both':
        #get the icd9 codes and descs from both sources, NZ and bp
        list_icd9_NZ, list_icd9_long_tit_NZ, map = umls2ICD9FromCSV(UMLS_code,default_path=default_path,map=map,onto_source='NZ',exact_or_narrower_only=exact_or_narrower_only)
        list_icd9_bp, list_icd9_long_tit_bp, map = umls2ICD9FromCSV(UMLS_code,default_path=default_path,map=map,onto_source='bp',exact_or_narrower_only=exact_or_narrower_only)
        #union the two lists from the two sources
        list_icd9 = list_icd9_NZ + [icd9 for icd9 in list_icd9_bp if icd9 not in list_icd9_NZ]
        list_icd9_long_tit = list_icd9_long_tit_NZ + [icd9_long_tit for icd9_long_tit in list_icd9_long_tit_bp if icd9_long_tit not in list_icd9_long_tit_NZ]
        return list_icd9, list_icd9_long_tit, map
    #print('from umls2ICD9FromCSV:', icd9_tmp)
    #print(icd9_long_tit_tmp)
    
    #extract icd9 codes and descs using regular expression
    pattern = "\"(.*?)\"|'(.*?)'"#"'(.*?)'" # here have two subgroups, as each element may be surrounded by either single quotes or double quotes
    list_icd9_tmp = re.findall(pattern,icd9_tmp)
    list_icd9_tmp = [x or y for x, y in list_icd9_tmp] #put the output tuple in the list into one
    #print(list_icd9_tmp)
    list_icd9_long_tit_tmp = re.findall(pattern,icd9_long_tit_tmp)
    list_icd9_long_tit_tmp = [x or y for x, y in list_icd9_long_tit_tmp]
    
    #remove beginning and ending white spaces - also remove the dot if it exists in the code.
    list_icd9_tmp = [icd9DotRemoval(icd9.strip()) for icd9 in list_icd9_tmp]
    list_icd9_long_tit_tmp = [icd9_long_tit.strip() for icd9_long_tit in list_icd9_long_tit_tmp]
    #get unique lists
    list_icd9_tmp = uniqueList(list_icd9_tmp)
    list_icd9_long_tit_tmp = uniqueList(list_icd9_long_tit_tmp)
    return list_icd9_tmp, list_icd9_long_tit_tmp, map

#input format: (i) here the ICD9_code does not contain dot(.). 
#              (ii) Also input the path whether the code matching sheet is stored
#              (iii) map initialised as None to store the coding matching map
#              (iv) the onto matching source, NZ (default, ordo-*icd10-icd9* with MoH NZ) or bp (ORDO-*UMLS-icd9* with bioportal ICD-9-CM), or both (from both NZ and bp)
#output: a tuple (list of ORDO IDs, list of ORDO_pref_labels)
def ICD92ORDOListFromCSV(ICD9_code,default_path='',map=None,onto_source='NZ',exact_or_narrower_only=False):
    if map is None:
        map = pd.read_excel(os.path.join(default_path,'./ontology/ORDO2UMLS_ICD10_ICD9+titles_final_v3.xlsx'),engine='openpyxl')
    # set onto_source to 'NZ' as default if not recognisable (not NZ or bp)   
    onto_source = 'NZ' if onto_source != 'NZ' and onto_source != 'bp' and onto_source != 'both' else onto_source    
    if exact_or_narrower_only and onto_source == 'NZ': 
        # update the source of NZ to exact or narrower only
        onto_source = 'NZ-E-N'
    if onto_source == 'both':
        #print('from both sources')
        list_ordo_ID_NZ, list_ordo_pref_label_NZ, map = ICD92ORDOListFromCSV(ICD9_code,default_path=default_path,map=map,onto_source='NZ',exact_or_narrower_only=exact_or_narrower_only)
        list_ordo_ID_bp, list_ordo_pref_label_bp, map = ICD92ORDOListFromCSV(ICD9_code,default_path=default_path,map=map,onto_source='bp',exact_or_narrower_only=exact_or_narrower_only)
        #union the two lists from the two sources
        list_ordo_ID = list_ordo_ID_NZ + [ordo_ID for ordo_ID in list_ordo_ID_bp if ordo_ID not in list_ordo_ID_NZ]
        list_ordo_pref_label = list_ordo_pref_label_NZ + [ordo_pref_label for ordo_pref_label in list_ordo_pref_label_bp if ordo_pref_label not in list_ordo_pref_label_NZ]
        return list_ordo_ID, list_ordo_pref_label, map
    # get icd9 column name with the onto_source
    icd9_code_source_cl_name = 'icd9-%s' % onto_source
    # add the heading white space for ICD9_code if using NZ matching
    ICD9_code = ' ' + ICD9_code.strip() if onto_source == 'NZ' or onto_source == 'NZ-E-N' else ICD9_code.strip()
    matched_df=map[map[icd9_code_source_cl_name].str.replace('.','').str.contains('\'%s\'' % ICD9_code)] # will this match to multiple rows? - yes
    if len(matched_df) == 0:
        #directly return empty lists if no matching
        return [],[], map
    ordo_ID_tmp = matched_df['ORDO ID'].to_string(index=False)
    ordo_pref_label_tmp = matched_df['Preferred Label'].to_string(index=False)
    #return ordo_ID_tmp, ordo_pref_label_tmp
    list_ordo_ID = [ordo_ID.strip()[26:] for ordo_ID in ordo_ID_tmp.split('\n')]
    list_ordo_pref_label = [ordo_pref_label.strip() for ordo_pref_label in ordo_pref_label_tmp.split('\n')]
    return list_ordo_ID, list_ordo_pref_label, map

# unused (this was later substituted by the function of ICD92ORDOListFromCSV)
# input: an ICD9 code, which does not contianing dot(.); and the path whether the code matching sheet is stored
# output: whether the ICD9 code has a linkage to the ORDO through ICD9 -> ICD10 -> ORDO  
def hasICD9linkage2ORDO(ICD9_code,default_path=''):
    map = pd.read_excel(os.path.join(default_path,'./ontology/ORDO2UMLS_ICD10_ICD9+titles_final_v3.xlsx'),engine='openpyxl',sheet_name="full sheet")
    matched_df=map[map['icd9'].str.contains('\' %s\'' % ICD9_code.strip())] # will this match to multiple rows? - yes
    return len(matched_df) > 0

# change the icd9 code format: with_dot to no_dot version   
# simply remove the dot from the code version
def icd9DotRemoval(icd9_code_with_dot):
    icd9_code_no_dot = ''.join(icd9_code_with_dot.split('.'))
    return icd9_code_no_dot

# reformat icd9 to dot version: add dot to icd9 code
# is_diag: should know whether it is a diagnosis code or a procedure code
# then just add the code to the right position
# acknowlegement to the function from https://github.com/jamesmullenbach/caml-mimic/blob/44a47455070d3d5c6ee69fb5305e32caec104960/datasets.py#L207    
def icd9DotAdd(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits, 
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code
    
# return results from the URL from the SPARQL endpoint
def get_results_from_sparql_url(SPARQL_url):
    response = requests.get(SPARQL_url)
    #print('response:',response)
    if response.status_code == 200:
        entity_data = response.json()
        return entity_data['results']['bindings']
    else:
        return ""
    
def isNotGroupOfDisorders(OrphanetID):
    ''' The SPARQL endpoint query
    prefix obolib:<http://purl.obolibrary.org/obo/>

    select distinct ?nodeID

    where {
    ?nodeID owl:annotatedSource <http://www.orpha.net/ORDO/Orphanet_101953>;
         owl:annotatedProperty <http://www.w3.org/2000/01/rdf-schema#subClassOf>;
         owl:annotatedTarget <http://www.orpha.net/ORDO/Orphanet_377794>;
         obolib:ECO_0000218 "Group of phenomes"@en.
    }

    LIMIT 1000
    '''
    ''' The URL:    https://www.orpha.net/sparql?default-graph-uri=&query=prefix+obolib%3A%3Chttp%3A%2F%2Fpurl.obolibrary.org%2Fobo%2F%3E%0D%0A%0D%0Aselect+distinct+%3FnodeID%0D%0A%0D%0Awhere+%7B%0D%0A%3FnodeID+owl%3AannotatedSource+%3Chttp%3A%2F%2Fwww.orpha.net%2FORDO%2FOrphanet_791%3E%3B%0D%0A+++++owl%3AannotatedProperty+%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F01%2Frdf-schema%23subClassOf%3E%3B%0D%0A+++++owl%3AannotatedTarget+%3Chttp%3A%2F%2Fwww.orpha.net%2FORDO%2FOrphanet_377794%3E%3B%0D%0A+++++obolib%3AECO_0000218+%22Group+of+phenomes%22%40en.%0D%0A%7D%0D%0A%0D%0ALIMIT+1000&should-sponge=&format=application%2Fsparql-results%2Bjson&timeout=0&debug=on
    '''
    
    # construct URL
    SPARQL_url = "https://www.orpha.net/sparql?default-graph-uri=&query=prefix+obolib%3A%3Chttp%3A%2F%2Fpurl.obolibrary.org%2Fobo%2F%3E%0D%0A%0D%0Aselect+distinct+%3FnodeID%0D%0A%0D%0Awhere+%7B%0D%0A%3FnodeID+owl%3AannotatedSource+%3Chttp%3A%2F%2Fwww.orpha.net%2FORDO%2F" + OrphanetID +  "%3E%3B%0D%0A+++++owl%3AannotatedProperty+%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F01%2Frdf-schema%23subClassOf%3E%3B%0D%0A+++++owl%3AannotatedTarget+%3Chttp%3A%2F%2Fwww.orpha.net%2FORDO%2FOrphanet_377794%3E%3B%0D%0A+++++obolib%3AECO_0000218+%22Group+of+phenomes%22%40en.%0D%0A%7D%0D%0A%0D%0ALIMIT+1000&should-sponge=&format=application%2Fsparql-results%2Bjson&timeout=0&debug=on"
    SPARQL_results = get_results_from_sparql_url(SPARQL_url)
    #print(len(SPARQL_results),SPARQL_results)
    return False if len(SPARQL_results) > 0 else True

def isDisease(OrphanetID):
    '''
    prefix obolib:<http://purl.obolibrary.org/obo/>

    select distinct ?nodeID

    where {
    ?nodeID owl:annotatedSource <http://www.orpha.net/ORDO/Orphanet_399>;
         owl:annotatedProperty <http://www.w3.org/2000/01/rdf-schema#subClassOf>;
         owl:annotatedTarget <http://www.orpha.net/ORDO/Orphanet_377788>;
         obolib:ECO_0000218 "Disease"@en.
    }

    LIMIT 1000
    '''
    # construct URL
    SPARQL_url = "https://www.orpha.net/sparql?default-graph-uri=&query=prefix+obolib%3A%3Chttp%3A%2F%2Fpurl.obolibrary.org%2Fobo%2F%3E%0D%0A%0D%0Aselect+distinct+%3FnodeID%0D%0A%0D%0Awhere+%7B%0D%0A%3FnodeID+owl%3AannotatedSource+%3Chttp%3A%2F%2Fwww.orpha.net%2FORDO%2F" + OrphanetID + "%3E%3B%0D%0A+++++owl%3AannotatedProperty+%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F01%2Frdf-schema%23subClassOf%3E%3B%0D%0A+++++owl%3AannotatedTarget+%3Chttp%3A%2F%2Fwww.orpha.net%2FORDO%2FOrphanet_377788%3E%3B%0D%0A+++++obolib%3AECO_0000218+%22Disease%22%40en.%0D%0A%7D%0D%0A%0D%0ALIMIT+1000&should-sponge=&format=application%2Fsparql-results%2Bjson&timeout=0&debug=on"
    SPARQL_results = get_results_from_sparql_url(SPARQL_url)
    #print(len(SPARQL_results),SPARQL_results)
    return True if len(SPARQL_results) > 0 else False

def getSupClassFromLeaf(OrphanetID):
    '''
    select distinct ?nodeID ?supclass

    where {
        <http://www.orpha.net/ORDO/Orphanet_166282> rdfs:subClassOf ?nodeID.
        ?nodeID owl:onProperty <http://purl.obolibrary.org/obo/BFO_0000050>;
                owl:someValuesFrom ?supclass.    
    }

    LIMIT 1
    '''
    # construct URL
    SPARQL_url = "https://www.orpha.net/sparql?default-graph-uri=&query=select+distinct+%3FnodeID+%3Fsupclass%0D%0A%0D%0Awhere+%7B%0D%0A++++%3Chttp%3A%2F%2Fwww.orpha.net%2FORDO%2F" + OrphanetID + "%3E+rdfs%3AsubClassOf+%3FnodeID.%0D%0A++++%3FnodeID+owl%3AonProperty+%3Chttp%3A%2F%2Fpurl.obolibrary.org%2Fobo%2FBFO_0000050%3E%3B%0D%0A++++++++++++owl%3AsomeValuesFrom+%3Fsupclass.++++%0D%0A%7D%0D%0A%0D%0ALIMIT+1&should-sponge=&format=application%2Fsparql-results%2Bjson&timeout=0&debug=on"
    SPARQL_results = get_results_from_sparql_url(SPARQL_url)
    #print(len(SPARQL_results),SPARQL_results)
    if len(SPARQL_results) == 0:
        return ''
    supClassUrl = SPARQL_results[0]['supclass']['value']
    supClassID = supClassUrl[len('http://www.orpha.net/ORDO/'):]
    print(supClassID)
    return supClassID

def getSupClassFromClass(OrphanetID):
    '''
    select distinct ?concept ?label

where {
    <http://www.orpha.net/ORDO/Orphanet_101934> rdfs:subClassOf ?concept.
    ?concept rdfs:label ?label
}

LIMIT 1000'''
    if OrphanetID == '':
        return ''
    # construct URL
    SPARQL_url = "https://www.orpha.net/sparql?default-graph-uri=&query=select+distinct+%3Fconcept+%3Flabel%0D%0A%0D%0Awhere+%7B%0D%0A++++%3Chttp%3A%2F%2Fwww.orpha.net%2FORDO%2F" + OrphanetID + "%3E+rdfs%3AsubClassOf+%3Fconcept.%0D%0A++++%3Fconcept+rdfs%3Alabel+%3Flabel%0D%0A%7D%0D%0A%0D%0ALIMIT+1000&should-sponge=&format=application%2Fsparql-results%2Bjson&timeout=0&debug=on"
    SPARQL_results = get_results_from_sparql_url(SPARQL_url)
    #print(len(SPARQL_results),SPARQL_results)
    if len(SPARQL_results) == 0:
        return OrphanetID
    supClassUrl = SPARQL_results[0]['concept']['value']
    supClassID = supClassUrl[len('http://www.orpha.net/ORDO/'):]
    print(supClassID)
    return supClassID

def get_ORDO_pref_label_with_dict(OrphanetID,dict_ORDO_to_pref_label,default_path='',map=None):
    if OrphanetID not in dict_ORDO_to_pref_label:
        #ORDO_pref_label = get_ORDO_pref_label(OrphanetID)
        ORDO_pref_label, map = get_ORDO_pref_label_from_CSV(OrphanetID,default_path,map)
        dict_ORDO_to_pref_label[OrphanetID] = ORDO_pref_label
        return ORDO_pref_label,dict_ORDO_to_pref_label,map
    else:
        return dict_ORDO_to_pref_label[OrphanetID], dict_ORDO_to_pref_label,map

#to do
def get_ORDO_pref_label_from_CSV(OrphanetID,default_path='',map=None):
    if OrphanetID == '':
        return '', map
    if map is None:
        map = pd.read_excel(os.path.join(default_path,'./ontology/ORDO2UMLS_ICD10_ICD9+titles_final_v3.xlsx'),engine='openpyxl')
    matched_df=map[map['ORDO ID'] == ('http://www.orpha.net/ORDO/' + OrphanetID)] # will this match to multiple rows? - yes
    if len(matched_df) == 0:
        #directly return empty lists if no matching
        return '', map
    ordo_pref_label_tmp = matched_df['Preferred Label'].to_string(index=False).strip()
    return ordo_pref_label_tmp, map
    
def get_ORDO_pref_label(OrphanetID):
    '''
select distinct ?label

where {
    <http://www.orpha.net/ORDO/Orphanet_218436> rdfs:label ?label
}

LIMIT 1'''
    if OrphanetID == '':
        return ''
    # construct URL
    SPARQL_url = "https://www.orpha.net/sparql?default-graph-uri=&query=select+distinct+%3Flabel%0D%0A%0D%0Awhere+%7B%0D%0A++++%3Chttp%3A%2F%2Fwww.orpha.net%2FORDO%2F" + OrphanetID + "%3E+rdfs%3Alabel+%3Flabel%0D%0A%7D%0D%0A%0D%0ALIMIT+1&should-sponge=&format=application%2Fsparql-results%2Bjson&timeout=0&debug=on"
    SPARQL_results = get_results_from_sparql_url(SPARQL_url)
    #print(len(SPARQL_results),SPARQL_results)
    if len(SPARQL_results) == 0:
        return OrphanetID
    ORDO_pref_label = SPARQL_results[0]['label']['value']    
    return ORDO_pref_label 
    
def uniqueList(input_list):
    used = set()
    unique_in_list = [x for x in input_list if x not in used and (used.add(x) or True)]
    return unique_in_list

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

def union(lst1,lst2):
    lst3 = lst1 + [value for value in lst2 if value not in lst1]
    return lst3
    
if __name__ == "__main__":
    pd.set_option("display.max_colwidth", 10000) # allowing showing and matching to long sequence
    
    #print(ordo2icd10FromJSON('http://www.orpha.net/ORDO/Orphanet_3325',exact_or_narrower_only=False))
    #print(get_ORDO_pref_label_from_CSV('Orphanet_101953')[0])
    #getSupClassFromClass(getSupClassFromLeaf('Orphanet_166282'))
    #print(umls2icd9('C0020538'))
    #print(umls2icd9List_bp('C2063873')[:2])
    #print(umls2ICD9FromCSV('C0280788',onto_source='both',exact_or_narrower_only=False))
    #print(umls2ordo('C0035828'))
    #print(umls2prefLabel('C0035828'))
    #print(umls2prefLabel('C0020538'))
    #print(isNotGroupOfDisorders('Orphanet_791'))
    #print(isNotGroupOfDisorders('Orphanet_101953'))
    print(isNotGroupOfDisorders('Orphanet_418'))
    #print(isDisease('Orphanet_791'))
    #print(isDisease('Orphanet_101953'))
    #print(isDisease('Orphanet_399'))
    #print(umls2ordoFromCSV('C1860464'))
    #print(umls2ordoListFromCSV('C1860464'))
    #print(ICD92ORDOListFromCSV('75569',onto_source='NZ',exact_or_narrower_only=False))
    #print(hasICD9linkage2ORDO('2881'))
    #print(uniqueList([1,2,2,3]))
    #print(icd9DotRemovalFmtChange('345.60'))