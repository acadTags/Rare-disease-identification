import requests
import pandas as pd
from collections import defaultdict
import re
#import constants

def get_rare_disease_umls():
    #df = pd.read_excel("ORDO2UMLS_ICD10_ICD9+titles.xlsx", sheet_name="full sheet")
    df = pd.read_excel("./ontology/ORDO2UMLS_ICD10_ICD9+titles_final_v3.xlsx",engine='openpyxl')
    # get the column of UMLS IDs

    dict_umls = defaultdict(int)
    pattern = "'(.*?)'"
    for i, row in df.iterrows():
        umlss = re.findall(pattern,row['UMLS IDs'])
        for j,k in enumerate(umlss):
            umlss[j] = k[5:]
            dict_umls[umlss[j]] = dict_umls[umlss[j]] + 1
        #df2.at[i, 'umlss'] = umlss
        
    #df2.to_excel('results_ver4.xlsx', index=False)
    #dict_umls_list = list(dict_umls.keys())
    #print(dict_umls_list, len(dict_umls_list))
    return dict_umls
    #list = ['C2063873', 'C1835813', 'C0035828', 'C0022577', 'C1853271', 'C0266393', 'C1836669', 'C0432307', 'C1840296', 'C1847902', 'C0265275', 'C1856051', 'C0311338', 'C1843691', 'C0349557', 'C1847896', 'C3266898', 'C0043008', 'C0037899', 'C0238506', 'C0542520', 'C1704429', 'C0342898', 'C0018553', 'C1853294', 'C0393559', 'C1862151', 'C3542021', 'C0917713', 'C1850100', 'C1866650', 'C0268641', 'C3494187', 'C0019069', 'C0000744', 'C0406735', 'C0406716', 'C0342788', 'C0152417', 'C1848922', 'C0039373', 'C0546999', 'C0796133', 'C1836123', 'C2931743', 'C2931645', 'C0002066', 'C1621958', 'C0017636', 'C0522624', 'C0033300', 'C0036161', 'C1857314', 'C0339512', 'C1845243', 'C0016782', 'C0266599', 'C1567744', 'C2931254', 'C0027877', 'C0175778', 'C2931648', 'C1861355', 'C1852201', 'C0265218', 'C0086873', 'C0271568', 'C2931836', 'C0265260', 'C1334968', 'C2239290', 'C0344993', 'C0265554', 'C0344760', 'C1167664', 'C0266642', 'C0949506', 'C1862840', 'C0265211', 'C1850808', 'C1833676', 'C0266190', 'C3489413', 'C0007361', 'C0238909', 'C1306663', 'C0265928', 'C0796081', 'C0687751', 'C0270952', 'C1850077', 'C2936797', 'C2931080', 'C0027832', 'C0027859', 'C1136041', 'C0271092', 'C0796005', 'C0263591', 'C1849928', 'C2931570', 'C2931571', 'C2931543', 'C0265339', 'C2931524', 'C0796151', 'C0334489', 'C0086647', 'C0342907', 'C0263417', 'C1321547', 'C1861451', 'C1845366', 'C0403553', 'C0272129', 'C0086774', 'C1370889', 'C0268416', 'C2751878', 'C0085576', 'C0393590', 'C1834558', 'C3850067', 'C0751731', 'C3544264', 'C1970011', 'C0152427', 'C1306837', 'C1837518', 'C1836929', 'C1845919', 'C1744559', 'C3502054', 'C0274888', 'C1869123', 'C0001627', 'C0409999', 'C0024307', 'C0796013', 'C1843075', 'C2931097', 'C1843330', 'C0752125', 'C0025183', 'C3696376', 'C0040588', 'C1849722', 'C2936827', 'C0019343', 'C0751540', 'C3854373', 'C1846055', 'C1300268', 'C1862103', 'C1328355', 'C0016781', 'C1842676', 'C0265970', 'C1839909', 'C1838912', 'C1833603', 'C1838781', 'C3714976', 'C0342883', 'C0751587', 'C0342544', 'C2239176', 'C0270970', 'C0023269', 'C1851920', 'C1318518', 'C2931285', 'C0936016', 'C0039585', 'C2720434', 'C0221355', 'C1849401', 'C0268569', 'C0039445', 'C0036202', 'C1867147', 'C0268621', 'C0038457', 'C0026850', 'C0268414', 'C0005859', 'C0032533', 'C1527406', 'C1845028', 'C0398791', 'C2930831', 'C0265499', 'C0268632', 'C3495554', 'C3495555', 'C1855348', 'C2673609', 'C0019202', 'C0220710', 'C1841972', 'C0546476', 'C2584774', 'C0263610', 'C0265255', 'C3888925', 'C0795817', 'C0280793', 'C0268126', 'C0014053', 'C0013264', 'C1867020', 'C2932714', 'C3888090', 'C1851945', 'C0013423', 'C2350875', 'C0006272', 'C3178805', 'C0153064', 'C1843225', 'C1963905', 'C1096902', 'C3279841', 'C0272362', 'C2700425', 'C1858302', 'C1842983', 'C0566602', 'C1859371', 'C0398692', 'C0205711', 'C0086649', 'C0585274', 'C0031069', 'C0339273', 'C0154773', 'C0206698', 'C0740277', 'C0393703', 'C1847839', 'C0699743', 'C0741296', 'C1399352', 'C0009677', 'C1850674', 'C1836899', 'C1855861', 'C0265934', 'C0340757', 'C0017075', 'C0010678', 'C1858501', 'C1838258', 'C0010674', 'C0036231', 'C0037929', 'C1861536', 'C0086652', 'C0406778', 'C1853984', 'C0270913', 'C1704375', 'C0342642', 'C0272238', 'C1838256', 'C0085810', 'C0038463', 'C0796031', 'C0796083', 'C0410207', 'C0752150', 'C1868508', 'C1302995', 'C1866777', 'C1854058', 'C0014850', 'C1838630', 'C0272118', 'C1970021', 'C2931142', 'C0268547', 'C1867443', 'C1849719', 'C1802405', 'C1846056', 'C1864689', 'C0345218', 'C1969623', 'C1860168', 'C2931087', 'C0266521', 'C0016751', 'C3887645', 'C0458219', 'C0152426', 'C2973787', 'C0034362', 'C0041234', 'C0345419', 'C0007965', 'C0472777', 'C1832399', 'C1855243', 'C0410530', 'C1853396', 'C1847720', 'C0751791', 'C0086648', 'C1275081', 'C2673885', 'C2931227', 'C1848863', 'C0344488', 'C0266470', 'C0263398', 'C0220669', 'C0079153', 'C1096116', 'C0796162', 'C0006181', 'C0152444', 'C1861301', 'C0039144', 'C1562689', 'C0796125', 'C0432263', 'C1853919', 'C2751683', 'C1567741', 'C0268238', 'C0409979', 'C1855229', 'C1855605', 'C1857532', 'C0242597', 'C0398738', 'C0687720', 'C0022972', 'C0022716', 'C0272302', 'C2717750', 'C1857069', 'C1623209', 'C2673198', 'C1843291', 'C0036980', 'C1846006', 'C0041341', 'C0020241', 'C1859081', 'C0152264', 'C0339540', 'C1840452', 'C1527231', 'C1844948', 'C0004712', 'C1970253', 'C0206638', 'C1832466', 'C0265966', 'C1322286', 'C0205969', 'C1274879', 'C0020256', 'C0152438', 'C0406817', 'C0039263', 'C0268274', 'C0339533', 'C1276035', 'C2676766', 'C0035021', 'C1720859', 'C0949595', 'C0685837', 'C3714581', 'C0345335', 'C0281508', 'C0685787', 'C0342669', 'C1866855', 'C1857316', 'C2676243', 'C1861963', 'C0024790', 'C0040021', 'C3501946', 'C1835265', 'C0334517', 'C0036220', 'C2673477', 'C0002895', 'C0024507', 'C0393725', 'C1266092', 'C3888099', 'C0019621', 'C0016065', 'C0152200', 'C0030327', 'C0206085', 'C0020630', 'C0265321', 'C2931326', 'C2931327', 'C2930918', 'C0014145', 'C0017547', 'C1855794', 'C0267663', 'C0796264', 'C0432268', 'C2677586', 'C1832334', 'C1867155', 'C0029454', 'C2750066', 'C0085077', 'C3163825', 'C1835171', 'C1843173', 'C0153065', 'C1865596', 'C0221018', 'C0265289', 'C0153579', 'C0238122', 'C1845861', 'C0268228', 'C3888317', 'C2931355', 'C0268125', 'C1112486', 'C0206717', 'C2931468', 'C1854023', 'C3711384', 'C0333693', 'C1867440', 'C0008928', 'C0342784', 'C0342773', 'C1838123', 'C0016395', 'C0263489', 'C0023647', 'C1863959', 'C0263390', 'C0878555', 'C0036920', 'C0265279', 'C0026703', 'C2931059', 'C0272350', 'C1849087', 'C0268563', 'C2363903']
    #return list 
    
def get_code_from_url(UMLS_code):
    dict_code = {}
    #construct SPARQL query
    #example SPARQL url http://linkedlifedata.com/sparql.json?query=PREFIX+skos%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2004%2F02%2Fskos%2Fcore%23%3E%0D%0APREFIX+rdfs%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F01%2Frdf-schema%23%3E%0D%0APREFIX+lld%3A+%3Chttp%3A%2F%2Flinkedlifedata.com%2Fresource%2F%3E%0D%0ASELECT+DISTINCT+%3Fconcept+%3Fcode+%3Fdescription%0D%0AWHERE+%7B%0D%0A%3Chttp%3A%2F%2Flinkedlifedata.com%2Fresource%2Fumls%2Fid%2FC0020538%3E+skos-xl%3AaltLabel+%3Fconcept.%0D%0A%3Fconcept+skos%3Anote+%22ICD-9-CM%22.%0D%0A%3Fconcept+skos%3Anotation+%3Fcode.%0D%0A%3Fconcept+skos-xl%3AliteralForm+%3Fdescription%0D%0A%7D%0D%0A&_implicit=false&implicit=true&_form=%2Fsparql
    SPARQL_url = "http://linkedlifedata.com/sparql.json?query=PREFIX+skos%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2004%2F02%2Fskos%2Fcore%23%3E%0D%0APREFIX+rdfs%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F01%2Frdf-schema%23%3E%0D%0APREFIX+lld%3A+%3Chttp%3A%2F%2Flinkedlifedata.com%2Fresource%2F%3E%0D%0ASELECT+DISTINCT+%3Fconcept+%3Fcode+%3Fdescription%0D%0AWHERE+%7B%0D%0A%3Chttp%3A%2F%2Flinkedlifedata.com%2Fresource%2Fumls%2Fid%2F" + UMLS_code + "%3E+skos-xl%3AaltLabel+%3Fconcept.%0D%0A%3Fconcept+skos%3Anote+%22ICD-9-CM%22.%0D%0A%3Fconcept+skos%3Anotation+%3Fcode.%0D%0A%3Fconcept+skos-xl%3AliteralForm+%3Fdescription%0D%0A%7D%0D%0A&_implicit=false&implicit=true&_form=%2Fsparql"
    #print(SPARQL_url)
    response = requests.get(SPARQL_url)
    #print(response.status_code)    
    if response.status_code == 200:
        entity_data = response.json()
        #print('entity_data',entity_data)
        for entry in entity_data["results"]["bindings"]:
            #print(entry["code"]["value"])
            code = entry["code"]["value"]
            if dict_code.get(code, None) != None:
                dict_code[code]=dict_code[code]+1
            else:
                dict_code[code]=1
    else:
        print(UMLS_code, 'querying failed:', response.status_code)
    return dict_code
    #";".join(dict_code.keys())

#print(get_code_from_url("C0020538"))

#adapted from https://thispointer.com/how-to-merge-two-or-more-dictionaries-in-python/
def mergeDict(dict1, dict2):
   ''' Merge dictionaries and sum values of common keys in list'''
   dict3 = {**dict1, **dict2}
   for key, value in dict3.items():
       if key in dict1 and key in dict2:
               dict3[key] = value + dict1[key]
 
   return dict3

if __name__ == '__main__':
    # to generate UMLS list matched to rare diseases in ORDO as the file umls_rare_diseases_final.txt
    from sent_bert_emb_viz_util import output_to_file
    
    dict_rare_dis_umls = get_rare_disease_umls()
    rare_dis_umls_str_by_lines = '\n'.join(list(dict_rare_dis_umls.keys()))
    output_to_file('umls_rare_diseases_final.txt',rare_dis_umls_str_by_lines)