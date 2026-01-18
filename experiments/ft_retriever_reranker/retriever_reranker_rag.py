import sys
sys.path.append('../')
from hf_auto_open import open_json

from prettytable import PrettyTable


def em(rag_list, model, union_method):
    total = 0
    for idx in range(len(rag_list)):
        retriever_q = rag_list[idx]
        q_id = retriever_q['id']
        if retriever_q['predicted_answer'][model][f'{q_id}_{union_method}'] in retriever_q['answers']:
            total += 1
    return round(total / len(rag_list), 4)


def evaluate(retriever, reranker, union_method):
    rag_list = open_json(f'ft_retriever_reranker/{retriever}/{reranker}/rag.json')

    columns = ['reranker']
    row = [reranker]

    # EM
    for model in ['llama-32-1b', 'gemma-3-4b', 'qwen-3-8b']:
        column = f'EM of {model}'
        val = em(rag_list, model, union_method)
        columns.append(column)
        row.append(val)

    return columns, row


if __name__ == '__main__':
    retrievers = ['dpr', 'colbert']
    rerankers = ['lit5dist-LiT5-Distill-base-v2', 'lit5dist-LiT5-Distill-large-v2', 'lit5dist-LiT5-Distill-xl-v2',
                 'monot5-monot5-3b-msmarco-10k', 'monot5-monot5-base-msmarco-10k', 'monot5-monot5-large-msmarco-10k',
                 'rankgpt-Llama-3.1-8B-Instruct', 'rankgpt-Qwen2.5-7B', 'rankt5-rankt5-3b', 'rankt5-rankt5-base',
                 'rankt5-rankt5-large',
                 'upr-gpt2-large', 'upr-t0-3b', 'upr-t5-large', 'upr-t5-small']
    union_methods = ['un_1', 'uf_1', 'un_3', 'uf_3', 'un_5', 'uf_5']
    for retriever in retrievers:
        for union_method in union_methods:
            table = PrettyTable()
            print(f'Evaluating {retriever}-{union_method}')
            for reranker in rerankers:
                columns, row = evaluate(retriever, reranker, union_method)
                table.field_names = columns
                table.add_row(row)
            print(table)
            print()
