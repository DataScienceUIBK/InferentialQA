import sys
sys.path.append('../')
from hf_auto_open import open_json

from prettytable import PrettyTable


def load_merge_file():
    merge_file = open_json("baselines/baseline.json")
    return merge_file


def hit_k(retriever_list, k=1):
    total = 0
    for idx in range(len(retriever_list)):
        retriever_q = retriever_list[idx]
        merged_q = merge_file[idx]

        retriever_q_ctxs_labels = [merged_q['ctxs'][ctx['id']]['label'] for ctx in retriever_q['reranked_ctxs']][:k]
        if retriever_q_ctxs_labels.count(2) + retriever_q_ctxs_labels.count(1) > 0:
            total += 1
    return round(total / len(retriever_list), 4)


def precision_k(retriever_list, k=1):
    total = 0
    for idx in range(len(retriever_list)):
        retriever_q = retriever_list[idx]
        merged_q = merge_file[idx]

        retriever_q_ctxs_labels = [merged_q['ctxs'][ctx['id']]['label'] for ctx in retriever_q['reranked_ctxs']][:k]
        total += ((retriever_q_ctxs_labels.count(2) + retriever_q_ctxs_labels.count(1)) / k)
    return round(total / len(retriever_list), 4)


def mrr(retriever_list, k):
    total = 0
    for idx in range(len(retriever_list)):
        retriever_q = retriever_list[idx]
        merged_q = merge_file[idx]

        retriever_q_ctxs_labels = [merged_q['ctxs'][ctx['id']]['label'] for ctx in retriever_q['reranked_ctxs']][:k]
        rank_first_rel = 0
        for rank, ret_ctx in enumerate(retriever_q_ctxs_labels, 1):
            if ret_ctx == 2 or ret_ctx == 1:
                rank_first_rel = rank
                break
        if rank_first_rel > 0:
            total += (1 / rank_first_rel)
    return round(total / len(retriever_list), 4)


def evaluate(retriever, reranker):
    retriever_list = open_json(f'baselines_reranker/wiki/{retriever}/{reranker}/reranker.json')

    columns = ['reranker']
    row = [reranker]

    # Hit@K
    print(f'Hit {reranker}')
    for k in [1, 5, 10, 20, 50, 100]:
        column = f'hit@{k}'
        val = hit_k(retriever_list, k)
        columns.append(column)
        row.append(val)

    # Precision@K
    print(f'Precision {reranker}')
    for k in [1, 5, 10, 20, 50, 100]:
        column = f'precision@{k}'
        val = precision_k(retriever_list, k)
        columns.append(column)
        row.append(val)

    # MRR@K
    print(f'MRR {reranker}')
    for k in [50, 100]:
        column = f'mrr@{k}'
        val = mrr(retriever_list, k)
        columns.append(column)
        row.append(val)

    return columns, row


if __name__ == '__main__':
    merge_file = load_merge_file()

    retrievers = ['bm25', 'dpr', 'colbert', 'contriever', 'bge']

    rerankers = ['lit5dist-LiT5-Distill-base-v2', 'lit5dist-LiT5-Distill-large-v2', 'lit5dist-LiT5-Distill-xl-v2',
                 'monot5-monot5-3b-msmarco-10k', 'monot5-monot5-base-msmarco-10k', 'monot5-monot5-large-msmarco-10k',
                 'rankgpt-Llama-3.1-8B-Instruct', 'rankgpt-Qwen2.5-7B', 'rankt5-rankt5-3b', 'rankt5-rankt5-base',
                 'rankt5-rankt5-large',
                 'upr-gpt2-large', 'upr-t0-3b', 'upr-t5-large', 'upr-t5-small']
    for retriever in retrievers:
        print(f'Evaluating {retriever}')
        table = PrettyTable()
        for reranker in rerankers:
            columns, row = evaluate(retriever, reranker)
            table.field_names = columns
            table.add_row(row)

        print(table)
        print()
