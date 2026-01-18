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


def evaluate(rerank, union_method):
    rag_list = open_json(f'ft_reranker/{rerank}_positive/rag.json')

    columns = ['rerank']
    row = [rerank]

    # EM
    for model in ['llama-32-1b', 'gemma-3-4b', 'qwen-3-8b']:
        column = f'EM of {model}'
        val = em(rag_list, model, union_method)
        columns.append(column)
        row.append(val)

    return columns, row


if __name__ == '__main__':
    rerankers = ['1', '5', '10', '50', '100']
    union_methods = ['un_1', 'uf_1', 'un_3', 'uf_3', 'un_5', 'uf_5']

    for union_method in union_methods:
        table = PrettyTable()
        print(f'Evaluating {union_method}')
        for rerank in rerankers:
            columns, row = evaluate(rerank, union_method)
            table.field_names = columns
            table.add_row(row)

        print(table)
        print()
        print()
