import sys
sys.path.append('../')
from hf_auto_open import open_json

import json
from prettytable import PrettyTable


def load_index_file():
    index_json_list = [json.loads(line) for line in open_json('index/test/retriever.jsonl')]
    return_list = []
    for q in index_json_list:
        label_dict = dict()
        for ctx in q['contexts']:
            label_dict[ctx['id']] = ctx['label']
        return_list.append(label_dict)
    return return_list


def hit_k(index_list, retriever_list, k, label_threshold=1, ignore_no_rels=True):
    hits = 0
    counted = 0
    for i in range(len(retriever_list)):
        gold = index_list[i]
        gold_rels = {cid for cid, lbl in gold.items() if lbl > label_threshold}

        if not gold_rels and ignore_no_rels:
            continue

        retrieved = [c['id'] for c in retriever_list[i]['ctxs']][:k]
        if any(cid in gold_rels for cid in retrieved):
            hits += 1
        counted += 1
    return round(hits / counted, 4) if counted > 0 else 0.0

def precision_k(index_list, retriever_list, k):
    total = 0.0
    for idx in range(len(retriever_list)):
        gold = index_list[idx]
        gold_rels = {cid for cid, lbl in gold.items() if lbl > 1}
        retrieved = [ctx['id'] for ctx in retriever_list[idx]['ctxs']][:k]
        denom = max(1, min(k, len(retrieved)))
        total += len(gold_rels.intersection(retrieved)) / denom
    return round(total / len(retriever_list), 4)


def mrr(index_list, retriever_list, k):
    total = 0.0
    for idx in range(len(retriever_list)):
        gold = index_list[idx]
        gold_rels = {cid for cid, lbl in gold.items() if lbl > 1}
        retrieved = [ctx['id'] for ctx in retriever_list[idx]['ctxs']][:k]
        rank_first_rel = 0
        for rank, cid in enumerate(retrieved, 1):
            if cid in gold_rels:
                rank_first_rel = rank
                break
        if rank_first_rel:
            total += 1.0 / rank_first_rel
    return round(total / len(retriever_list), 4)


def evaluate(retriever):
    retriever_list = open_json(f'retriever/{retriever}/retriever.json')
    index_list = load_index_file()

    columns = ['retriever']
    row = [retriever]

    # Recall@K
    print(f'Hit {retriever}')
    for k in [1, 5, 10, 20, 50, 100]:
        column = f'hit@{k}'
        val = hit_k(index_list, retriever_list, k)
        columns.append(column)
        row.append(val)

    # Precision@K
    print(f'Precision {retriever}')
    for k in [1, 5, 10, 20, 50, 100]:
        column = f'precision@{k}'
        val = precision_k(index_list, retriever_list, k)
        columns.append(column)
        row.append(val)

    # MRR@K
    print(f'MRR {retriever}')
    for k in [50, 100]:
        column = f'mrr@{k}'
        val = mrr(index_list, retriever_list, k)
        columns.append(column)
        row.append(val)

    return columns, row


if __name__ == '__main__':
    retrievers = ['bm25', 'dpr', 'colbert', 'contriever', 'bge']

    table = PrettyTable()
    for retriever in retrievers:
        columns, row = evaluate(retriever)
        table.field_names = columns
        table.add_row(row)

    print(table)
    print()