import sys
sys.path.append('../')
from hf_auto_open import open_json

import json
import ir_datasets


def load_our_dataset():
    corpus = [json.loads(line)['contents'] for line in open_json('index/corpus.jsonl')]
    return corpus


def load_msmarco():
    dataset = ir_datasets.load("msmarco-passage")
    corpus = [doc.text for doc in dataset.docs_iter()]
    return corpus


def load_wiki():
    dataset = ir_datasets.load("dpr-w100")
    corpus = [doc.text for doc in dataset.docs_iter()]
    return corpus


def compute_avg_length(corpus):
    return sum([len(psg.split(' ')) for psg in corpus]) / len(corpus)


def main():
    corpuses = {'our_dataset': load_our_dataset, 'msmarco': load_msmarco, 'wiki': load_wiki}
    for key, val in corpuses.items():
        corpus = val()
        avg_length = compute_avg_length(corpus)
        print(key, avg_length)


if __name__ == '__main__':
    main()
