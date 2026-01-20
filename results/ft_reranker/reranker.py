import sys
sys.path.append('../')
from hf_auto_open import open_json

import json
import math
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


def ndcg(index_list, retriever_list, k):
    total = 0.0
    for idx in range(len(retriever_list)):
        gold = index_list[idx]  # {cid: grade}
        retrieved = [ctx['id'] for ctx in retriever_list[idx]['reranked_ctxs']][:k]

        dcg = 0.0
        for r, cid in enumerate(retrieved, 1):
            rel = gold.get(cid, 0)
            if rel > 0:
                dcg += (2 ** rel - 1) / math.log2(r + 1)

        ideal = sorted((lbl for lbl in gold.values() if lbl > 0), reverse=True)[:k]
        idcg = sum((2 ** rel - 1) / math.log2(r + 1) for r, rel in enumerate(ideal, 1))

        ndcg_q = (dcg / idcg) if idcg > 0 else 0.0
        total += ndcg_q
    return round(total / len(retriever_list), 4)

def evaluate(reranker):
    retriever_list = open_json(f'ft_reranker/{reranker}_positive/reranker.json')
    index_list = load_index_file()

    columns = ['reranker']
    row = [reranker]

    # NDCG@K
    print(f'NDCG {reranker}')
    for k in [1, 5, 10, 20, 50, 100]:
        column = f'ndcg@{k}'
        val = ndcg(index_list, retriever_list, k)
        columns.append(column)
        row.append(val)

    return columns, row

if __name__ == '__main__':
    rerankers = ['1', '5', '10', '50', '100']

    table = PrettyTable()
    for reranker in rerankers:
        columns, row = evaluate(reranker)
        table.field_names = columns
        table.add_row(row)

    print(table)
    print()
