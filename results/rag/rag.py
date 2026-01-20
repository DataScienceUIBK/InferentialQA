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
            label_dict[ctx['id']] = (ctx['rag_answers'], q['answers'], ctx['label'])
        return_list.append(label_dict)
    return return_list


def acc(index_list, relevant_label, model):
    total = 0
    relevant_counter = 0
    for idx in range(len(index_list)):
        index_q = index_list[idx]
        for ctx_id in index_q.keys():
            if str(index_q[ctx_id][2]) in relevant_label:
                relevant_counter += 1
                if index_q[ctx_id][0][model] in index_q[ctx_id][1]:
                    total += 1
    return round(total / relevant_counter, 4), relevant_counter


def em(index_list, relevant_label, model):
    total = 0
    for idx in range(len(index_list)):
        index_q = index_list[idx]
        for ctx_id in index_q.keys():
            if str(index_q[ctx_id][2]) in relevant_label:
                if index_q[ctx_id][0][model] in index_q[ctx_id][1]:
                    total += 1
                    break
    return round(total / len(index_list), 4)


def evaluate(relevant_label):
    index_list = load_index_file()

    columns = ['relevant docs']
    row = [relevant_label]

    # Acc
    for model in ['llama-32-1b', 'gemma-3-4b', 'qwen-3-8b']:
        column = f'Accuracy of {model}'
        val, num_of_rels = acc(index_list, relevant_label, model)
        columns.append(column)
        row.append(val)
    print(f'{num_of_rels}: {325 * len(index_list)} ({round(num_of_rels / (325 * len(index_list)), 4)})')

    # em
    for model in ['llama-32-1b', 'gemma-3-4b', 'qwen-3-8b']:
        column = f'EM of {model}'
        val = em(index_list, relevant_label, model)
        columns.append(column)
        row.append(val)

    return columns, row


if __name__ == '__main__':
    relevant_labels = ['1 and 2', '1', '2']

    table = PrettyTable()
    for relevant_label in relevant_labels:
        print(f'Evaluating {relevant_label}')
        columns, row = evaluate(relevant_label)
        table.field_names = columns
        table.add_row(row)

    print(table)
    print()
