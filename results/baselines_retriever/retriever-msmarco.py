import sys
sys.path.append('../')
from hf_auto_open import open_json

from prettytable import PrettyTable


def load_merge_file():
    merge_file = open_json(f'baselines/baseline.json')
    return merge_file


def hit_k(retriever_list, k=1):
    total = 0
    for idx in range(len(retriever_list)):
        retriever_q = retriever_list[idx]
        merged_q = merge_file[idx]

        retriever_q_ctxs_labels = [merged_q['ctxs'][ctx['id']]['label'] for ctx in retriever_q['ctxs']][:k]
        if retriever_q_ctxs_labels.count(2) + retriever_q_ctxs_labels.count(1) > 0:
            total += 1
    return round(total / len(retriever_list), 4)


def precision_k(retriever_list, k=1):
    total = 0
    for idx in range(len(retriever_list)):
        retriever_q = retriever_list[idx]
        merged_q = merge_file[idx]

        retriever_q_ctxs_labels = [merged_q['ctxs'][ctx['id']]['label'] for ctx in retriever_q['ctxs']][:k]
        total += ((retriever_q_ctxs_labels.count(2) + retriever_q_ctxs_labels.count(1)) / k)
    return round(total / len(retriever_list), 4)


def mrr(retriever_list, k):
    total = 0
    for idx in range(len(retriever_list)):
        retriever_q = retriever_list[idx]
        merged_q = merge_file[idx]

        retriever_q_ctxs_labels = [merged_q['ctxs'][ctx['id']]['label'] for ctx in retriever_q['ctxs']][:k]
        rank_first_rel = 0
        for rank, ret_ctx in enumerate(retriever_q_ctxs_labels, 1):
            if ret_ctx == 2 or ret_ctx == 1:
                rank_first_rel = rank
                break
        if rank_first_rel > 0:
            total += (1 / rank_first_rel)
    return round(total / len(retriever_list), 4)


def evaluate(retriever):
    retriever_list = open_json(f'baselines_retriever/msmarco/{retriever}/retriever.json')

    columns = ['retriever']
    row = [retriever]

    # Recall@K
    print(f'Hit {retriever}')
    for k in [1, 5, 10, 20, 50, 100]:
        column = f'hit@{k}'
        val = hit_k(retriever_list, k)
        columns.append(column)
        row.append(val)

    # Precision@K
    print(f'Precision {retriever}')
    for k in [1, 5, 10, 20, 50, 100]:
        column = f'precision@{k}'
        val = precision_k(retriever_list, k)
        columns.append(column)
        row.append(val)

    # MRR@K
    print(f'MRR {retriever}')
    for k in [50, 100]:
        column = f'mrr@{k}'
        val = mrr(retriever_list, k)
        columns.append(column)
        row.append(val)

    return columns, row


if __name__ == '__main__':
    merge_file = load_merge_file()

    retrievers = ['bm25', 'dpr', 'colbert', 'contriever', 'bge']

    table = PrettyTable()
    for retriever in retrievers:
        columns, row = evaluate(retriever)
        table.field_names = columns
        table.add_row(row)

    print(table)
    print()
