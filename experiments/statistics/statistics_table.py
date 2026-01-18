import sys
sys.path.append('../')
from hf_auto_open import open_json

import json
import os
from tqdm import tqdm


def load_our_dataset():
    retrievers = dict()
    if os.path.exists('./our_dataset.json'):
        with open('./our_dataset.json', 'r') as f:
            retrievers = json.load(f)
    else:
        rets = os.listdir('../__hf_cache/retriever')
        for ret in rets:
            ret_key = ret.split('_')[0]
            retrievers[ret_key] = []
            for ret in tqdm(rets):
                data = open_json(f'retriever/{ret}/retriever.json')
                for q in data:
                    question = q['question']
                    for psg in q['ctxs'][:100]:
                        retrievers[ret_key].append({'question': question, 'passage': psg['text']})
        with open('./our_dataset.json', 'w') as f:
            json.dump(retrievers, f, indent=4)
    return retrievers


def load_msmarco():
    retrievers = dict()
    if os.path.exists('./msmarco.json'):
        with open('./msmarco.json', 'r') as f:
            retrievers = json.load(f)
    else:
        rets = os.listdir('../__hf_cache/baselines_retriever/msmarco')
        for ret in rets:
            ret_key = ret.split('_')[0]
            retrievers[ret_key] = []
            for ret in tqdm(rets):
                data = open_json(f'baselines_retriever/msmarco/{ret}/retriever.json')
                for q in data:
                    question = q['question']
                    for psg in q['ctxs']:
                        retrievers[ret_key].append({'question': question, 'passage': psg['text']})
        with open('./msmarco.json', 'w') as f:
            json.dump(retrievers, f, indent=4)
    return retrievers


def load_wikipedia():
    retrievers = dict()
    if os.path.exists('./wikipedia.json'):
        with open('./wikipedia.json', 'r') as f:
            retrievers = json.load(f)
    else:
        rets = os.listdir('../__hf_cache/baselines_retriever/wiki')
        for ret in rets:
            ret_key = ret.split('_')[0]
            retrievers[ret_key] = []
            for ret in tqdm(rets):
                data = open_json(f'baselines_retriever/wiki/{ret}/retriever.json')
                for q in data:
                    question = q['question']
                    for psg in q['ctxs']:
                        retrievers[ret_key].append({'question': question, 'passage': psg['text']})
        with open('./wikipedia.json', 'w') as f:
            json.dump(retrievers, f, indent=4)
    return retrievers


def compute_avg_length(corpus):
    length = 0
    count = 0
    for retrirver in corpus:
        psgs = corpus[retrirver]
        for psg in psgs:
            length += len(psg['passage'].split(' '))
            count += 1
    return length / count


def main():
    corpuses = {'our_dataset': load_our_dataset, 'msmarco': load_msmarco, 'wiki': load_wikipedia}
    for key, val in corpuses.items():
        corpus = val()
        avg_length = compute_avg_length(corpus)
        print(key, avg_length)


if __name__ == '__main__':
    main()
