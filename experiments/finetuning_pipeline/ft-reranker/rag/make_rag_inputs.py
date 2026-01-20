import json
import os
import shutil
from tqdm import tqdm


def union_norm(hints_list):
    final_list = []
    for hints in hints_list:
        final_list = list(dict.fromkeys(final_list + hints))
    final_context = ' '.join(list(final_list))
    return final_context


def union_freq(hints_list):
    alpha = 0.6
    beta = 0.4
    sent_dict = dict()
    for rank, hints in enumerate(hints_list, 1):
        for pos, hint in enumerate(hints, 1):
            if hint not in sent_dict:
                sent_dict[hint] = []
            sent_dict[hint].append((rank, pos))
    final_dict = dict()
    for hint in sent_dict:
        rank_list, pos_list = zip(*sent_dict[hint])
        reciprocal_rank_part = sum([1 / rank for rank in rank_list])
        position_part = sum([1 / pos for pos in pos_list])
        final_dict[hint] = alpha * reciprocal_rank_part + beta * position_part
    final_dict = dict(sorted(final_dict.items(), key=lambda x: x[1], reverse=True))
    final_context = ' '.join(list(final_dict.keys()))
    return final_context


def generate_top_k(contexts, k):
    ctxs_k = contexts[:k]
    hints_list = []
    for ctx in ctxs_k:
        hint_orders = [int(rank) for rank in ctx['id'].split('_')[-1]]
        q_id = ctx['title']
        hints = [full_dataset[q_id]['hints'][rank]['hint'] for rank in hint_orders]
        hints_list.append(hints)
    union_norm_context = union_norm(hints_list)
    union_freq_context = union_freq(hints_list)
    return union_norm_context, union_freq_context


def clean(retriever, reranker):
    with open(f'../retriever/{retriever}/finetuned_monot5_10_positive_{retriever}.json', 'r') as f:
        json_data = json.load(f)
    for fd_q_id, j_q in zip(test_set.keys(), json_data):
        ctxs = []
        for k in [1, 3, 5]:
            union_norm_context, union_freq_context = generate_top_k(j_q['reranked_ctxs'], k)
            ctxs.append({'id': f'{fd_q_id}_un_{k}', 'text': union_norm_context})
            ctxs.append({'id': f'{fd_q_id}_uf_{k}', 'text': union_freq_context})
        j_q['ctxs'] = ctxs
        del j_q['reranked_ctxs']
        j_q['id'] = fd_q_id
    save_path = f'./cleaned/rerankers/{retriever}'
    os.makedirs(save_path, exist_ok=True)
    save_path = f'{save_path}/{reranker}.json'
    with open(save_path, 'w') as f:
        json.dump(json_data, f, indent=4)


if __name__ == '__main__':
    with open('../../../dataset/test.json', 'r') as f:
        test_set = {rec['id']: rec for rec in json.load(f)}
    with open('../../../dataset/dev.json', 'r') as f:
        dev_set = {rec['id']: rec for rec in json.load(f)}
    with open('../../../dataset/train.json', 'r') as f:
        train_set = {rec['id']: rec for rec in json.load(f)}

    full_dataset = dict()
    full_dataset.update(test_set)
    full_dataset.update(dev_set)
    full_dataset.update(train_set)

    retrievers = ['bm25', 'colbert', 'dpr', 'contriever', 'bge']
    for retriever in tqdm(retrievers):
        clean(retriever, 'ft_monot5')
