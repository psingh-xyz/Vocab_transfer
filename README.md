# Thesis

### Description:

data_utils.py: defines all dataset used

Step1: Prepare tokenization. 

Step2: Pretrain the model. 

Step3: create the matched vocabulary file

Step4: mapping of old tokens with the new one

Step5: Pretrain the model on the new dataset

Step6: finetune the model

### Steps to follow:

1. python .\vocabulary_transfer_of_medical_text\scripts\step1.py --dataset .\datasets\data1_en_wiki\en_wiki.txt --prefix .\output\step1\en_wiki_8k --tok-type
unigram --vocab_size 8000

2. python .\vocabulary_transfer_of_medical_text\scripts\step1.py --dataset .\datasets\data2_en_medical\train.dat --prefix .\output\step1\en_medical_8k --tok-type unigram --vocab_size 8000

3. python .\vocabulary_transfer_of_medical_text\scripts\step1.py --dataset .\datasets\data3_ohsumed\train.txt --prefix .\output\step1\ohsumed_8k --tok-type unigram --vocab_size 8000

4. python .\vocabulary_transfer_of_medical_text\scripts\step2.py --dataset_pth .\datasets\data1_en_wiki\en_wiki.txt --snp_path .\output\step1\en_wiki_8k.model --save_path ./output/step2/models/rawBert_rawTokenizer/

5.  python .\vocabulary_transfer_of_medical_text\scripts\step3.py --wiki_vocab .\output\step1\en_wiki_8k.vocab --task_vocab .\output\step1\en_medical_8k.vocab --out_vocab .\output\step3\wiki_medical_8k --matcher 1

6.  python .\vocabulary_transfer_of_medical_text\scripts\step3.py --wiki_vocab .\output\step1\en_wiki_8k.vocab --task_vocab .\output\step1\en_medical_8k.vocab --out_vocab .\output\step3\wiki_medical_8k --matcher 2

7.  python .\vocabulary_transfer_of_medical_text\scripts\step3.py --wiki_vocab .\output\step1\en_wiki_8k.vocab --task_vocab .\output\step1\ohsumed_8k.vocab --out_vocab .\output\step3\wiki_ohsumed_8k --matcher 1

8.  python .\vocabulary_transfer_of_medical_text\scripts\step3.py --wiki_vocab .\output\step1\en_wiki_8k.vocab --task_vocab .\output\step1\ohsumed_8k.vocab --out_vocab .\output\step3\wiki_ohsumed_8k --matcher 2

9. python .\vocabulary_transfer_of_medical_text\scripts\step4.py --mapping_file_1 .\output\step3\wiki_ohsumed_8k_matcher_f1.tsv --mapping_file_2 .\output\step3\wiki_ohsumed_8k_matcher_f2.tsv --source_bert_model .\output\step2\models\rawBert_rawTokenizer\Bert_Wiki_8k --save_pht .\output\step4\wiki_ohsumed_8k\

10. python .\vocabulary_transfer_of_medical_text\scripts\step4.py --mapping_file_1 .\output\step3\wiki_medical_8k_matcher_f1.tsv --mapping_file_2 .\output\step3\wiki_medical_8k_matcher_f2.tsv --source_bert_model .\output\step2\models\rawBert_rawTokenizer\Bert_Wiki_8k --save_pht .\output\step4\wiki_medical_8k\
    
11. python .\vocabulary_transfer_of_medical_text\scripts\step5.py --dataset_name quora_dataset --experiment_folder avg --sp .\output\step1\en_medical_8k.model --experiments_dir wiki_medical_8k --root_dir .\output\step4\

12. python .\vocabulary_transfer_of_medical_text\scripts\step5.py --dataset_name quora_dataset --experiment_folder random --sp .\output\step1\en_medical_8k.model --experiments_dir wiki_medical_8k --root_dir .\output\step4\

13. python .\vocabulary_transfer_of_medical_text\scripts\step6.py --dataset_name quora_dataset --experiment_folder random --sp_model_pth .\output\step1\en_medical_8k.model --experiments_dir .\output\step4\wiki_medical_8k_2mlm --save_dir
 .\output\step6
