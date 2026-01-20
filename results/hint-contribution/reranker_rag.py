import sys
sys.path.append('../')
from hf_auto_open import open_json

import json
from prettytable import PrettyTable

def assess_num_of_hints(rag_list, retriever_list, model):
    num_of_hints_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for idx in range(len(rag_list)):
        rag_q = rag_list[idx]
        retriever_q = retriever_list[idx]

        q_id = rag_q['id']
        if rag_q['predicted_answer'][model][f'{q_id}_un_1'] in rag_q['answers']:
            _id = retriever_q['reranked_ctxs'][0]['id']
            num_of_hints = _id.split('_')[3].count('1')
            num_of_hints_dict[num_of_hints] += 1
    return num_of_hints_dict


def evaluate_num_of_hints(retriever):
    rag_list = open_json(f'reranker/{retriever}/rag.json')
    retriever_list = open_json(f'reranker/{retriever}/reranker.json')

    columns = ['reranker']
    row = [retriever]

    # EM
    for model in ['llama-32-1b', 'gemma-3-4b', 'qwen-3-8b']:
        val = assess_num_of_hints(rag_list, retriever_list, model)
        sum_val = sum(val.values())
        for key, value in val.items():
            column = f'hints@{model}-{key}'
            columns.append(column)
            row.append(round(value / sum_val, 4))

    return columns, row


if __name__ == '__main__':
    rerankers = ['upr-gpt2-large']
    for retriever in rerankers:
        print(f'Evaluating {retriever}')
        table = PrettyTable()
        columns, row = evaluate_num_of_hints(retriever)
        table.field_names = columns
        table.add_row(row)

        print(table)
        print()
