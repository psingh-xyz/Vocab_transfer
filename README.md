# Thesis

### Description:
data_utils.py: defines all dataset used
Step1: Prepare tokenization. 
Step2: Pretrain the model. 
Step3: cretae the matched voacbulary file
Step4: mapping of old tokens with the new one
Step5: Pretrain the model on the new dataset
Step6: finetune the model

### Steps to follow:
1.run common.py file
2.In data_utils file: use en_wiki dataset and medical text dataset and run
3.run step1 as it is and in command line provide the vocab size using code: !python /content/thesis/step1.py  --dataset /content/thesis/en_wiki.txt --prefix wiki_8k --tok-type unigram --vocab_size 8000
from this step two file will be obtained. 1. wiki_8k.model 2. wiki_8k.vocab
4.run step1 for medical_text also. code: !python /content/thesis/step1.py  --dataset /content/thesis/medical/train.dat --prefix medical_8k --tok-type unigram --vocab_size 8000
from this step two file will be obtained. 1. medical_8k.model 2. medical_8k.vocab
5.for step2 in argument parsing --snp_path add the file path of wiki_8k.model obtained from step1. From this step a bert model folder will be obtained as output.
6.for step3 add the wiki_8k.vocab and medica_8k.vocab file path obtained from step1. run the step3 two times. 1 parser.add_argument('--matcher', type=int, default=1) and default=2.two output file of vocab matcher will be obtained as _f1.tsv and _f2.tsv
7. for step4 and the two tsv file obtained from step 3 and bert model path obtained from step2.

