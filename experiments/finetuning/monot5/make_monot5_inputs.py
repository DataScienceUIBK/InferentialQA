import json
import os
import random as rnd
from tqdm import tqdm

rnd.seed(42)


def generate_train_data(num_of_passages):
    records = []
    with open('../../index/train/retriever.jsonl', 'r') as f:
        jsonl_file = f.readlines()
    with open('../../dataset/train.json', 'r') as f:
        id_json = json.load(f)

    jsonl_list = []
    for line in jsonl_file:
        query_dict = json.loads(line)
        jsonl_list.append(query_dict)

    for idx, (_id_rec, _) in tqdm(enumerate(zip(id_json, jsonl_list)), total=len(jsonl_list),
                                  desc=f'Generating train data for {num_of_passages} passages'):
        query_id = _id_rec['id']
        query = _id_rec['question']
        positive_passages = rnd.sample(jsonl_list[idx]['contexts'], num_of_passages)
        for pp_idx, positive_rec in enumerate(positive_passages):
            positive_content = {'docid': positive_rec['id'], 'title': query_id, 'text': positive_rec['text'].replace('\t', ' ')}

            valid_negative_queries = list(range(0, len(jsonl_list)))
            valid_negative_queries.remove(idx)
            negative_query_index = rnd.choice(valid_negative_queries)
            negative_rec = jsonl_list[negative_query_index]['contexts'][rnd.randint(1, 324)]
            negative_content = {'docid': negative_rec['id'], 'title': id_json[negative_query_index]['id'],
                                'text': negative_rec['text'].replace('\t', ' ')}
            records.append(
                [query, positive_content['text'], negative_content['text']])
    rnd.shuffle(records)
    os.makedirs(f'./{num_of_passages}_positive', exist_ok=True)
    with open(f'./{num_of_passages}_positive/train_data.tsv', 'w') as f:
        for record in records:
            f.write('\t'.join(record)+'\n')

def generate_test_data(num_of_passages):
    cleaned_test = []
    with open('../../index/test/retriever.jsonl', 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]
    for q in lines:
        question = q['question']
        answers = q['answers']
        ctxs = []
        for ctx in q['contexts']:
            p_id = ctx['id']
            text = ctx['text']
            ctxs.append({'score': 1.0, 'has_answer': False, 'id': p_id, 'text': text, 'title': None})
        cleaned_test.append({'question': question, 'answers': answers, 'ctxs': ctxs})

    with open(f'./{num_of_passages}_positive/test_data.json', 'w') as f:
        json.dump(cleaned_test, f, indent=4)

def generate(number_of_passages):
    for num_of_psg in number_of_passages:
        generate_train_data(num_of_psg)
        generate_test_data(num_of_psg)


if __name__ == '__main__':
    generate([1, 5, 10, 50, 100])
