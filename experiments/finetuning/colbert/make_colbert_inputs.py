import json
import os
import random as rnd
from tqdm import tqdm

rnd.seed(42)


def generate_collection(index_file):
    id_map = dict()
    if index_file == 'test':
        with open(f'../../index/corpus.jsonl', 'r') as f:
            jsonl_file = f.readlines()
    else:
        with open(f'../../index/train/index.jsonl', 'r') as f:
            jsonl_file = f.readlines()

    records = []
    for _idx, line in tqdm(enumerate(jsonl_file), total=len(jsonl_file),
                           desc=f'Generating collection of {index_file} set'):
        ctx_dict = json.loads(line)
        ctx_id = ctx_dict['id']
        ctx_content = ctx_dict['contents']
        if ctx_id not in id_map:
            id_map[ctx_id] = _idx
        query_line = "\t".join([str(_idx), ctx_content])
        records.append(query_line)

    os.makedirs('./corpus_dir', exist_ok=True)
    with open(f'./corpus_dir/collection_{index_file}.tsv', 'w') as f:
        for record in records:
            f.write(record + "\n")

    if index_file == 'test':
        with open(f'./corpus_dir/id_map_{index_file}.json', 'w') as f:
            id_map = {v: k for k, v in id_map.items()}
            json.dump(id_map, f, indent=4)

    return id_map


def generate_query(index_file, num_of_passages):
    id_map = dict()
    records = []

    with open(f'../../index/{index_file}/retriever.jsonl', 'r') as f:
        jsonl_file = f.readlines()

    if index_file == 'test':
        hintrag_path = '../../dataset/test.json'
    else:
        hintrag_path = '../../dataset/train.json'
    with open(hintrag_path, 'r') as f:
        id_json = json.load(f)

    jsonl_list = []
    for line in jsonl_file:
        query_dict = json.loads(line)
        jsonl_list.append(query_dict)
    for idx, (_id_rec, q) in tqdm(enumerate(zip(id_json, jsonl_list)), total=len(jsonl_list),
                                  desc=f'Generating query of {index_file} set for {num_of_passages} passages'):
        query_id = _id_rec['id']
        query = q['question']
        if query_id in id_map:
            raise Exception('Query IDs are not unique.')
        id_map[query_id] = []
        for rep_idx in range(idx * num_of_passages, (idx + 1) * num_of_passages):
            id_map[query_id].append(rep_idx)
            records.append('\t'.join([str(rep_idx), query]))

    base_path = f'./{num_of_passages}_positive' if index_file == 'train' else '.'
    os.makedirs(f'{base_path}/dev_dir', exist_ok=True)
    with open(f'{base_path}/dev_dir/queries_{index_file}.tsv', 'w') as f:
        for record in records:
            f.write(record + "\n")

    if index_file == 'test':
        with open(f'{base_path}/dev_dir/id_map_{index_file}.json', 'w') as f:
            id_map = {v[0]: k for k, v in id_map.items()}
            json.dump(id_map, f, indent=4)

    return id_map


def generate_train_data(num_of_passages, collection_id_map, query_id_map):
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
        positive_passages = rnd.sample(jsonl_list[idx]['contexts'], num_of_passages)
        for pp_idx, positive_rec in enumerate(positive_passages):
            positive_content = {'docid': positive_rec['id'], 'title': query_id, 'text': positive_rec['text']}

            valid_negative_queries = list(range(0, len(jsonl_list)))
            valid_negative_queries.remove(idx)
            negative_query_index = rnd.choice(valid_negative_queries)
            negative_rec = jsonl_list[negative_query_index]['contexts'][rnd.randint(1, 324)]
            negative_content = {'docid': negative_rec['id'], 'title': id_json[negative_query_index]['id'],
                                'text': negative_rec['text']}
            records.append(
                [int(query_id_map[query_id][pp_idx]), int(collection_id_map[positive_content['docid']]),
                 int(collection_id_map[negative_content['docid']])])
    rnd.shuffle(records)
    os.makedirs(f'./{num_of_passages}_positive/train_dir', exist_ok=True)
    with open(f'./{num_of_passages}_positive/train_dir/train_data.jsonl', 'w') as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def generate(number_of_passages):
    generate_collection('test')
    collection_id_map = generate_collection('train')

    generate_query('test', 1)

    for num_of_psg in number_of_passages:
        query_id_map = generate_query('train', num_of_psg)
        generate_train_data(num_of_psg, collection_id_map, query_id_map)


if __name__ == '__main__':
    generate([1, 5, 10, 50, 100, 200])
