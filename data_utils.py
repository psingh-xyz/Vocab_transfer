import os

from torch.utils.data import DataLoader, Dataset
import torch
from typing import List
from tqdm import tqdm
import pandas as pd
from datasets import load_from_disk
import re
from bs4 import BeautifulSoup


class BPEDropout:
    def __init__(self, sp, alpha):
        self.sp = sp
        self.alpha = alpha

    def encode_as_ids(self, text):
        return self.sp.encode(text, enable_sampling=True, alpha=self.alpha)


class wikiADataset(Dataset):
    @staticmethod
    def output_sentences(path, save):
        texts = load_from_disk(path)['train'][:1000000]['text']
        filtered_texts = []
        for text in texts:
            parts = [part.strip() for part in re.split('_[A-Za-z]+_[A-Za-z]+_|_[A-Za-z]+_', text)]
            filtered_texts.append(" ".join(parts).strip())

        with open(save, 'w') as f:
            for text in filtered_texts:
                f.write(f'{text}\n')

    def __init__(self, sp, input_pth='/content/ChangeTokenization/wiki_en.txt', block_size=256):
        with open(input_pth) as f:
            data = f.readlines()
        self.examples = []

        for row in tqdm(data):
            tokenized_text = sp.encode_as_ids(row)

            if len(tokenized_text) > block_size:
                for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                    block = tokenized_text[i: i + block_size]
                    block = [4] + block + [5]
                    self.examples.append(block)
            else:
                self.examples.append([4] + tokenized_text + [5])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class SentimentMLMDataset(Dataset):
    @staticmethod
    def _get_texts(path, save):
        ds = load_from_disk(path)
        texts = ds['train']['text']

        with open(save, 'w') as f:
            for text in texts:
                f.write(f'{text}\n')

    def __init__(self, path, sp, block_size=256):
        df = load_from_disk(path)
        data = df['train']
        self.examples = []

        for row in tqdm(data):
            tokenized_text = sp.encode_as_ids(row['text'])
            if len(tokenized_text) < 3:
                continue
            if len(tokenized_text) > block_size:
                for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                    block = tokenized_text[i:i + block_size]
                    block = [4] + block + [5]
                    self.examples.append(block)
            else:
                self.examples.append([4] + tokenized_text + [5])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class SentimentCLFDataset(Dataset):
    def __init__(self, path, sp, train=False):
        df = load_from_disk(path)
        if train:
            data = df['train']
        else:
            data = df['test']

        self.examples = []
        self.target = []

        for row in data:
            tokenized_text = sp.encode_as_ids(row['text'])
            if len(tokenized_text) > 412 or len(tokenized_text) < 3:
                continue

            self.examples.append([4] + tokenized_text + [5])
            self.target.append(row['sentiment'] // 4)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(int(self.target[i]))


class HyperpartisanMLMDataset(Dataset):
    @staticmethod
    def _get_texts(path, save):
        ds = load_from_disk(path)
        texts = ds['train']['text']

        with open(save, 'w') as f:
            for text in tqdm(texts):
                text = BeautifulSoup(text).text.replace('\n', ' ')
                f.write(f'{text}\n')

    def __init__(self, path, sp, block_size=256):
        df = load_from_disk(path)
        data = df['train']
        self.examples = []

        for row in tqdm(data):
            text = BeautifulSoup(row['text']).text.replace('\n', ' ')
            tokenized_text = sp.encode_as_ids(text)
            if len(tokenized_text) < 3:
                continue
            if len(tokenized_text) > block_size:
                for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                    block = tokenized_text[i:i + block_size]
                    block = [4] + block + [5]
                    self.examples.append(block)
            else:
                self.examples.append([4] + tokenized_text + [5])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class HyperpartisanCLFDataset(Dataset):
    def __init__(self, path, sp, train=False):
        df = load_from_disk(path)
        if train:
            data = df['train']
        else:
            data = df['validation']

        self.examples = []
        self.target = []

        for row in data:
            text = BeautifulSoup(row['text']).text.replace('\n', ' ')
            tokenized_text = sp.encode_as_ids(text)
            if len(tokenized_text) < 3:
                continue

            self.examples.append([4] + tokenized_text[:412] + [5])
            self.target.append(int(row['hyperpartisan']))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(int(self.target[i]))


class QuoraMLMDataset(Dataset):
    @staticmethod
    def _get_texts(path, save):
        df = pd.read_csv(path)
        df.question_text = df.question_text.fillna(" ")
        texts = df['question_text'][:1000000]

        with open(save, 'w') as f:
            for text in texts:
                f.write(f'{text}\n')

    def __init__(self, path, sp, block_size=256):
        df = pd.read_csv(f'{path}/train.csv')
        df.question_text = df.question_text.fillna(" ")
        data = df['question_text'][:1000000] #train
        self.examples = []
        self.target = []
        for row in tqdm(data):
            tokenized_text = sp.encode_as_ids(row)
            if len(tokenized_text) < 3:
                continue
            if len(tokenized_text) > block_size:
                for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                    block = tokenized_text[i:i + block_size]
                    block = [4] + block + [5]
                    self.examples.append(block)
            else:
                self.examples.append([4] + tokenized_text + [5])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class QuoraCLFDataset(Dataset):
    def __init__(self, path, sp, train=False):
        df = pd.read_csv(f'{path}/train.csv')
        df.question_text = df.question_text.fillna(" ")
        if train:
            data = df['question_text'].values[:1000000]
            labels = df['target'].values[:1000000]
        else:
            data = df['question_text'].values[1000000:]
            labels = df['target'].values[1000000:]
        self.examples = []
        self.target = []
        i = 0
        for _ in tqdm(range(len(data))):
            row = data[i]
            tokenized_text = sp.encode_as_ids(row)
            if len(tokenized_text) > 412 or len(tokenized_text) < 3:
                i += 1
                continue
            self.examples.append([4] + tokenized_text + [5])
            self.target.append(labels[i])
            i += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(int(self.target[i]))
      








class IllnessDataset(Dataset):
    @staticmethod
    def _read_and_tokenize(data, sp, block_size):
        texts = data[1]
        labels = (data[0] - 1).to_numpy()

        tokenized_texts = []
        for text in texts:
            tokenized_text = sp.encode_as_ids(text)[:block_size]
            tokenized_texts.append(tokenized_text)

        return tokenized_texts, labels

    @staticmethod
    def output_sentences(path, save):
        train_data = list(pd.read_csv(f'{path}/train.dat', delimiter='\t', header=None)[1])
        test_data = list(pd.read_csv(f'{path}/test.dat', delimiter='\t', header=None)[0])

        train_data.extend(test_data)
        with open(save, 'w') as f:
            for line in train_data:
                f.write(f'{line}\n')

    def __init__(self, sp, train, block_size=256):
        path = '/content/ChangeTokenization/dataset/train.dat'

        data = pd.read_csv(path, delimiter='\t', header=None)
        train_size = int(len(data) * 0.8)

        if train:
            data = data.iloc[:train_size]
        else:
            data = data.iloc[train_size:]

        tokenized_text, target = IllnessDataset._read_and_tokenize(data, sp, block_size)

        self.tokenized_text = tokenized_text
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, i):
        return torch.tensor(self.tokenized_text[i], dtype=torch.long), torch.tensor(int(self.target[i]))