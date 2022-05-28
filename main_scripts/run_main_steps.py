# aggregating the main steps of ontology-based and weakly supervised rare disease identification from clincal notes
# ruuning step1 requires SemEHR-processed output

from subprocess import Popen, STDOUT
import argparse
import time

start_time = time.time()
    
parser = argparse.ArgumentParser(description="an implementation of the pipeline for ontology-based and weakly supervised approach for rare disease identification")
parser.add_argument('-d','--dataset', type=str,
                help="dataset name", default='MIMIC-III')
parser.add_argument('-dc','--data-category', type=str,
                help="category of data", default='Discharge summary')
parser.add_argument('-st','--supervision-type', type=str,
                help="type of supervision: weak or strong", default='weak')
parser.add_argument('-train','--training', 
                    help='whether to train a weakly supervised phenotype confirmation model.)',action='store_true',default=False)
parser.add_argument('-test','--testing', 
                    help='whether test on evaluation sheet - the gold annotation is needed in advance',action='store_true',default=False)
parser.add_argument('-trans','--direct-transfer', 
                    help='whether to direct transfer a weakly supervised phenotype confirmation model.)',action='store_true',default=False)
parser.add_argument('-py3','--python-version3', 
                    help='whether or using python version 3 command.)',action='store_true',default=False)
args = parser.parse_args()

command_python = 'python3' if args.python_version3 else 'python'
if args.dataset == 'MIMIC-III':
    step1_cmd = command_python + ' step0_mimic3_data_processing.py -dc \"%s\"' % args.data_category
else:
    step1_cmd = command_python + ' step0.1_tayside_data_processing.py -d \"%s\"' % args.dataset
step2_cmd = command_python + ' step1_tr_data_creat_ment_disamb.py -d \"%s\" -dc \"%s\"' % (args.dataset,args.data_category) 
if args.training:
    step3_cmd = command_python + ' step3.4_train_and_test_model_ment_disamb_pred.py -d \"%s\" -dc  \"%s\" -f%s' % (args.dataset,args.data_category,' -t' if args.testing else '') 
else:
    step3_cmd = command_python + ' step3.6_test_model_ment_disamb_applied_for_rad.py -d \"%s\" -dc \"%s\" -st \"%s\" -f' % (args.dataset,args.data_category,args.supervision_type)
step4_cmd = command_python + ' step9_processing_all_documents.py -d \"%s\" -dc \"%s\" -st \"%s\" %s -en' % (args.dataset,args.data_category,args.supervision_type,'-trans' if args.direct_transfer else '')   

list_cmd = [step1_cmd,step2_cmd,step3_cmd,step4_cmd]

for ind, cmd in enumerate(list_cmd):
    print(cmd)
    start_time_tmp = time.time()
    p = Popen(cmd, shell=True, stderr=STDOUT)
    p.wait()

    if 0 != p.returncode:
        print('Command %s wrong!' % cmd)
        break
    else:
        print('Command %s completed successfully!' % cmd)
    print('step %d used:' % (ind+1), time.time() - start_time_tmp, "seconds")    

print('the whole program used:', time.time() - start_time, "seconds")    