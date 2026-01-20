import json
import os
import shutil
import random as rnd
from tqdm import tqdm

rnd.seed(42)


def generate_collection():
    with open('../../index/corpus.jsonl', 'r') as f:
        jsonl_file = f.readlines()

    records = []
    for line in tqdm(jsonl_file, desc=f'Generating collection'):
        query_dict = json.loads(line)
        query_dict = {'docid': query_dict['id'], 'text': query_dict['contents'], 'title': query_dict['title']}
        records.append(query_dict)

    os.makedirs('./corpus_dir', exist_ok=True)
    with open('./corpus_dir/corpus.jsonl', 'w') as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # wget.download(
    #     "https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus/resolve/main/msmarco-passage-corpus.py?download=true",
    #     "./corpus_dir/msmarco-passage-corpus.py")
    # shutil.copy('./msmarco-passage-corpus.py', './corpus_dir/msmarco-passage-corpus.py')


def generate_query():
    records = []

    with open('../../index/test/retriever.jsonl', 'r') as f:
        jsonl_file = f.readlines()
    with open('../../dataset/test.json', 'r') as f:
        id_json = json.load(f)

    jsonl_list = []
    for line in jsonl_file:
        query_dict = json.loads(line)
        jsonl_list.append(query_dict)
    for idx, (_id_rec, q) in tqdm(enumerate(zip(id_json, jsonl_list)), total=len(jsonl_list), desc=f'Generating query'):
        query_id = _id_rec['id']
        query = q['question']
        records.append({'query_id': query_id, 'query': query})

    os.makedirs('./dev_dir', exist_ok=True)
    with open('./dev_dir/dev_data.jsonl', 'w') as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # wget.download(
    #     "https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus/resolve/main/msmarco-passage-corpus.py?download=true",
    #     "./dev_dir/msmarco-passage-corpus.py")
    # shutil.copy('./msmarco-passage-corpus.py', './dev_dir/msmarco-passage-corpus.py')

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

    for idx, (_id_rec, q) in tqdm(enumerate(zip(id_json, jsonl_list)), total=len(jsonl_list),
                                  desc=f'Generating train data for {num_of_passages} passages'):
        query_id = _id_rec['id']
        query = q['question']
        positive_passages = rnd.sample(jsonl_list[idx]['contexts'], num_of_passages)
        for pp_idx, positive_rec in enumerate(positive_passages):
            positive_content = {'docid': positive_rec['id'], 'title': query_id, 'text': positive_rec['text']}

            valid_negative_queries = list(range(0, len(jsonl_list)))
            valid_negative_queries.remove(idx)
            negative_query_index = rnd.choice(valid_negative_queries)
            negative_rec = jsonl_list[negative_query_index]['contexts'][rnd.randint(1, 324)]
            negative_content = {'docid': negative_rec['id'], 'title': id_json[negative_query_index]['id'],
                                'text': negative_rec['text']}
            records.append({'query_id': f'{query_id}_{pp_idx}', 'query': query, 'positive_passages': [positive_content],
                            'negative_passages': [negative_content]})
    rnd.shuffle(records)
    os.makedirs(f'./{num_of_passages}_positive/train_dir', exist_ok=True)
    with open(f'./{num_of_passages}_positive/train_dir/train_data.jsonl', 'w') as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def generate(number_of_passages):
    generate_collection()
    generate_query()

    for num_of_psg in number_of_passages:
        generate_train_data(num_of_psg)


if __name__ == '__main__':
    generate([1, 5, 10, 50, 100, 200])
