import sys
sys.path.append('../')
from hf_auto_open import open_json

import json


def load_our_dataset():
    index_json_list = [json.loads(line) for line in open_json('index/test/retriever.jsonl')]
    return_list = []
    for q in index_json_list:
        label_dict = dict()
        for ctx in q['contexts']:
            label_dict[ctx['id']] = ctx['label']
        return_list.append(label_dict)
    return return_list


def load_wiki_dataset():
    index_json_list = open_json('baselines/baseline.json')
    return_list = []
    for q in index_json_list:
        label_dict = dict()
        for ctx in q['ctxs']:
            label_dict[ctx] = q['ctxs'][ctx]['label']
        return_list.append(label_dict)
    return return_list


def load_msmarco_dataset():
    index_json_list = open_json('baselines/baseline.json')
    return_list = []
    for q in index_json_list:
        label_dict = dict()
        for ctx in q['ctxs']:
            label_dict[ctx] = q['ctxs'][ctx]['label']
        return_list.append(label_dict)
    return return_list


if __name__ == '__main__':
    our_dataset = load_our_dataset()
    wiki_dataset = load_wiki_dataset()
    msmarco_dataset = load_msmarco_dataset()

    retrievers = ['bm25', 'dpr', 'colbert', 'contriever', 'bge']

    count = 0
    for retriever in retrievers:
        dataset_json = open_json(f'retriever/{retriever}/retriever.json')
        for idx, q in enumerate(dataset_json):
            for ctx in q['ctxs']:
                ctx_id = ctx['id']
                if ctx_id in our_dataset[idx]:
                    if our_dataset[idx][ctx_id] == 2:
                        count += 1
    print(f'Our dataset: {count} from {1728 * 100 * 5} -> ({round(count / (1728 * 100 * 5), 3)})')

    count = 0
    for retriever in retrievers:
        dataset_json = open_json(f'baselines_retriever/wiki/{retriever}/retriever.json')
        for idx, q in enumerate(dataset_json):
            for ctx in q['ctxs']:
                ctx_id = ctx['id']
                if wiki_dataset[idx][ctx_id] == 2:
                    count += 1
    print(f'Wikipedia: {count} from {1728 * 100 * 5} -> ({round(count / (1728 * 100 * 5), 3)})')

    count = 0
    for retriever in retrievers:
        dataset_json = open_json(f'baselines_retriever/msmarco/{retriever}/retriever.json')
        for idx, q in enumerate(dataset_json):
            for ctx in q['ctxs']:
                ctx_id = ctx['id']
                if wiki_dataset[idx][ctx_id] == 2:
                    count += 1
    print(f'MSMARCO: {count} from {1728 * 100 * 5} -> ({round(count / (1728 * 100 * 5), 3)})')
