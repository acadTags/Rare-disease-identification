# aggregating the published mention-level data to document-level data
# input: any published data generated by step9.5_data_rd_linking_publishing
# output: document-level data

import pandas as pd
#mention_level_data_fn = 'test_set_RD_ann_MIMIC_III_disch_sample-rad.csv'
mention_level_data_fn = 'full_set_RD_ann_MIMIC_III_disch_sample.csv'

gold = pd.read_csv(mention_level_data_fn)
gold = gold[gold["gold text-to-ORDO label"]==1]
gold["Gold Entities"] = list(gold[['mention offset in full document', 'mention','UMLS with desc','ORDO with desc']].itertuples(index=False, name=None))#(gold['mention offset in full document'],gold['mention'],gold['UMLS with desc'],gold['ORDO with desc'])
d = gold.groupby(["ROW_ID","SUBJECT_ID","HADM_ID"], as_index=False)["Gold Entities"].agg(lambda x: list(x))
d.to_excel(mention_level_data_fn[:len(mention_level_data_fn)-len('.csv')] + ' doc-level.xlsx',index=False)