import argparse        # module, parsing command-line arguments
import os              # module, interact with the operating system
import torch           # pytorch lib, tensor computations with GPU acceleration
from tqdm import tqdm  # extensible progress bar for iterating over loops
import sentencepiece as spm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification
from transformers import AdamW     # deep learning model architecture
from torch.utils.data import RandomSampler
from torch.nn.modules.linear import Linear
import time ### new
# import pandas as pd #### new

from common import check_dir
from data_utils_new import QuoraCLFDataset, SentimentCLFDataset, HyperpartisanCLFDataset, BPEDropout

data_path = "/content/thesis/Dataset/quora_dataset.csv"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, default= 'quora/quora_dataset.csv')
    parser.add_argument('--experiment_folder', default='init')  # default='random/'
    parser.add_argument('--sp_model_pth', default='/thesis/medical_8k/sp.model')
    parser.add_argument('--experiments_dir', default='/thesis/models/matched_models/wiki_medical_8k_2mlm/')
    parser.add_argument('--save_dir', default='/thesis/results/medical_8k/')
    parser.add_argument('--dataset_path', type=str, default= data_path)
    parser.add_argument('--num_epoch', type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    # Debug: Print the dataset path
    print(f"1 Dataset Path: {args.dataset_path}/{args.dataset_name}")

    data_inf = f"{time.time()}, {args.dataset_name}"  ### collect data ### new 
    # df_inf = pd.read_csv("inferences.csv") if os.path.exists("inferences.csv") else pd.DataFrame() ### existing df file or new df ### new
    # df_inf.at[len(df_inf.index), "dataset_name"] = args.dataset_name  ### store model name or some distinguishable name at last index ### new
    
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu

    DATASETS = {
        'quora_dataset': QuoraCLFDataset,
        #'sentiment': SentimentCLFDataset,
        'hyperpartisan_corrected': HyperpartisanCLFDataset
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    check_dir(args.save_dir)
    model_pth = os.path.join(args.experiments_dir, args.experiment_folder)
    model = BertForSequenceClassification.from_pretrained(model_pth, num_labels=2)

    model.to(device=device)

    print(f"model load: {model_pth}")

    sp = spm.SentencePieceProcessor()
    sp.load(args.sp_model_pth)
    print('SentencePiece size', sp.get_piece_size())

    if 'bpe_dropout' in args.experiments_dir:
        train_sp = BPEDropout(sp, 0.1)
    else:
        train_sp = sp

    BPEDataset_train = DATASETS[args.dataset_name](f'{args.dataset_path}/{args.dataset_name}', train_sp, train=True)
    print(f"2 Training Dataset size: {len(BPEDataset_train)}")  # Debug
    
    BPEDataset_dev = DATASETS[args.dataset_name](f'{args.dataset_path}/{args.dataset_name}', sp, train=False)
    print(f"3 Development Dataset size: {len(BPEDataset_dev)}")  # Debug
    
    BPEDataset_test = DATASETS[args.dataset_name](f'{args.dataset_path}/{args.dataset_name}', sp, train=False)
    print(f"4 Testing Dataset size: {len(BPEDataset_test)}")  # Debug

    def collate(examples):
        result_input, result_labels = [], []
        for inputs, labels in examples:
            result_input.append(inputs)
            result_labels.append(labels)
        result_input = pad_sequence(result_input, batch_first=True, padding_value=sp.pad_id())
        return result_input, torch.tensor(result_labels)
   

    train_dataloader = DataLoader(BPEDataset_train, batch_size=40, collate_fn=collate, shuffle=True)
    dev_dataloader = DataLoader(BPEDataset_dev, batch_size=16, collate_fn=collate, shuffle=True)
    test_dataloader = DataLoader(BPEDataset_test, batch_size=16, collate_fn=collate, shuffle=True)


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)

    model.zero_grad()
    epoch_train_acc = []
    epoch_train_loss = []
    train_loss = []

    epoch_test_acc = []
    epoch_test_loss = []
    test_loss = []

    epoch_dev_acc = []
    epoch_dev_loss = []
    dev_loss = []

    epoch_couter = 0
    for _ in range(args.num_epoch):
        print('epoch', epoch_couter)
        epoch_couter += 1
        model.train()
        tmp_epoch_train_loss = 0
        tmp_epoch_test_loss = 0

        tmp_epoch_train_acc = 0
        tmp_epoch_test_acc = 0

        tmp_epoch_dev_loss = 0
        tmp_epoch_dev_acc = 0

        for batch in tqdm(train_dataloader):
            inputs, labels = batch
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(inputs, labels=labels)
            loss = outputs[0]

            tmp_epoch_train_acc += (torch.argmax(outputs[1], dim=1) == labels).sum().data.cpu().numpy()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            train_loss.append(loss.item())
            tmp_epoch_train_loss += loss.item()

            optimizer.step()
            model.zero_grad()

        model.eval()
        with torch.no_grad():
            for batch in tqdm(dev_dataloader):
                inputs, labels = batch
                inputs = inputs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(inputs, labels=labels)
                loss = outputs[0]
                tmp_epoch_dev_acc += (torch.argmax(outputs[1], dim=1) == labels).sum().data.cpu().numpy()
                dev_loss.append(loss.item())
                tmp_epoch_dev_loss += loss.item()

            for batch in tqdm(test_dataloader):
                inputs, labels = batch
                inputs = inputs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(inputs, labels=labels)
                loss = outputs[0]
                tmp_epoch_test_acc += (torch.argmax(outputs[1], dim=1) == labels).sum().data.cpu().numpy()
                test_loss.append(loss.item())
                tmp_epoch_test_loss += loss.item()

                          # Measure inference time after training and validation are complete
            num_batches_to_measure = 10  # You can adjust this number
            start_time = time.time()
        with torch.no_grad():
          for i, batch in enumerate(test_dataloader):
            if i >= num_batches_to_measure:
              break
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            model(inputs, labels=labels)
            elapsed_time = time.time() - start_time
            print(f"Time taken for {num_batches_to_measure} batches: {elapsed_time:.3f} seconds")
          # df_inf.loc[len(df_inf.index) - 1, f"epoch_{epoch_couter}"] = elapsed_time  ### add the new record to a new column ### new
          data_inf += f", {elapsed_time}"   ### add more data ### new

        epoch_train_acc.append(100 * float(tmp_epoch_train_acc) / float(len(BPEDataset_train)))
        epoch_train_loss.append(tmp_epoch_train_loss / len(train_dataloader))

        epoch_test_acc.append(100 * float(tmp_epoch_test_acc) / float(len(BPEDataset_test)))
        epoch_test_loss.append(tmp_epoch_test_loss / len(test_dataloader))

        epoch_dev_acc.append(100 * float(tmp_epoch_dev_acc) / float(len(BPEDataset_dev)))
        epoch_dev_loss.append(tmp_epoch_dev_loss / len(dev_dataloader))

        stage = 'stage3' if 'mlm' in args.save_dir else 'stage2'
        check_dir(os.path.join(args.save_dir, args.experiment_folder))

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_batch_train_loss.txt'), 'w') as f:
            for item in train_loss:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_batch_test_loss.txt'), 'w') as f:
            for item in test_loss:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_batch_dev_loss.txt'), 'w') as f:
            for item in dev_loss:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_epoch_train_loss.txt'), 'w') as f:
            for item in epoch_train_loss:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_epoch_test_loss.txt'), 'w') as f:
            for item in epoch_test_loss:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_epoch_dev_loss.txt'), 'w') as f:
            for item in epoch_dev_loss:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_epoch_test_acc.txt'), 'w') as f:
            for item in epoch_test_acc:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_epoch_train_acc.txt'), 'w') as f:
            for item in epoch_train_acc:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_epoch_dev_acc.txt'), 'w') as f:
            for item in epoch_dev_acc:
                f.write("%s\n" % item)

    # df_inf.to_csv("inferences.csv", index=False)  ### save updated file ### new
    with open("inferences.csv", "a") as file:     ### store collected data
      file.write(f"{data_inf} \n")


if __name__ == '__main__':
    main()
