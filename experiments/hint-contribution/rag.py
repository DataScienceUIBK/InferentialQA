import sys
sys.path.append('../')
from hf_auto_open import open_json

import json
from prettytable import PrettyTable

def assess_num_of_hints(rag_list, model):
    num_of_hints_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for idx in range(len(rag_list)):
        rag_q = rag_list[idx]
        for ctx in rag_q['contexts']:
            ctx_id = ctx['id']
            rag_answer = ctx['rag_answers'][model]
            if rag_answer in rag_q['answers']:
                num_of_hints = ctx_id.split('_')[3].count('1')
                num_of_hints_dict[num_of_hints] += 1
    return num_of_hints_dict


def evaluate_num_of_hints():
    rag_list = [json.loads(line) for line in open_json(f'index/test/retriever.jsonl')]

    columns = ['reranker']
    row = ['Oracle']

    # EM
    for model in ['llama-32-1b', 'gemma-3-4b', 'qwen-3-8b']:
        val = assess_num_of_hints(rag_list, model)
        sum_val = sum(val.values())
        for key, value in val.items():
            column = f'hints@{model}-{key}'
            columns.append(column)
            row.append(round(value / sum_val, 4))

    return columns, row


if __name__ == '__main__':

    print(f'Evaluating Oracle')
    table = PrettyTable()
    columns, row = evaluate_num_of_hints()
    table.field_names = columns
    table.add_row(row)

    print(table)
    print()
