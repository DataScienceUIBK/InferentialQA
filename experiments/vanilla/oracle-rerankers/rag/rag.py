import os
import json


def load_index():
    index_dict = []
    with open('../../../index/test/retriever.jsonl', 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]
    for line in lines:
        q_dict = dict()
        for ctx in line['contexts']:
            q_dict[ctx['text']] = ctx['rag_answers']
        index_dict.append(q_dict)
    return index_dict


def rag(reranker, rag_model):
    os.makedirs(f"./qa_results/{reranker}", exist_ok=True)
    with open(f'./cleaned/rerankers/{reranker}.json', 'r') as f:
        reranker_json = json.load(f)
    qa_results = []
    for idx, q in enumerate(reranker_json):
        for ctx in q['ctxs']:
            ctx_id = ctx['id']
            ctx_text = ctx['text'].strip()
            qa_results.append([ctx_id, index_file[idx][ctx_text][rag_model]])
    with open(f"./qa_results/{reranker}/answers_{rag_model}.json", "w") as f:
        json.dump(qa_results, f, indent=4)


if __name__ == "__main__":
    index_file = load_index()
    rerankers = ['lit5dist-LiT5-Distill-base-v2', 'lit5dist-LiT5-Distill-large-v2', 'lit5dist-LiT5-Distill-xl-v2',
                 'monot5-monot5-3b-msmarco-10k', 'monot5-monot5-base-msmarco-10k', 'monot5-monot5-large-msmarco-10k',
                 'rankgpt-Llama-3.1-8B-Instruct', 'rankgpt-Qwen2.5-7B', 'rankt5-rankt5-3b', 'rankt5-rankt5-base',
                 'rankt5-rankt5-large',
                 'upr-gpt2-large', 'upr-t0-3b', 'upr-t5-large', 'upr-t5-small']
    rag_models = ['llama-32-1b', 'gemma-3-4b', 'qwen-3-8b']

    for reranker in rerankers:
        for rag_model in rag_models:
            rag(reranker, rag_model)
