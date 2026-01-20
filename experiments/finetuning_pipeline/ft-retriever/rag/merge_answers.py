import json
import os
from tqdm import tqdm

def merge(retriever, reranker):
    if reranker is None:
        with open(f'./cleaned/retrievers/{retriever}_output_1000.json', 'r') as f:
            json_data = json.load(f)
    else:
        with open(f'./cleaned/rerankers/{retriever}/{reranker}.json', 'r') as f:
            json_data = json.load(f)

    models_map = {'gemma-3-1b-it': 'gemma-3-1b',
                  'llama-3.2-1b-instruct': 'llama-32-1b',
                  'gemma-3-4b-it': 'gemma-3-4b',
                  'qwen3-4b': 'qwen-3-4b',
                  'llama-3.1-8b-instruct': 'llama-31-8b',
                  'qwen3-8b': 'qwen-3-8b'}

    for model in ['meta-llama/Llama-3.2-1B-Instruct', 'google/gemma-3-4b-it', 'Qwen/Qwen3-8B']:
        model_name = models_map[model.split('/')[1].lower()]

        with open(f'./qa_results/{retriever}/{reranker}/answers_{model_name}.json', 'r') as f:
            answer_json = {ans[0]: ans[1] for ans in json.load(f)}

        for q in json_data:
            q_id = q['id']
            if 'predicted_answer' not in q:
                q['predicted_answer'] = dict()
            # q['predicted_answer'][model_name] = {answer_json[q_id]}
            q['predicted_answer'][model_name] = dict()
            for rag_strategy in ['un_1', 'uf_1', 'un_3', 'uf_3', 'un_5', 'uf_5']:
                ctx_id = f'{q_id}_{rag_strategy}'
                q['predicted_answer'][model_name][ctx_id] = answer_json[ctx_id]

    os.makedirs(f'./rag_results/{retriever}', exist_ok=True)
    with open(f'./rag_results/{retriever}/{reranker}.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)


def main():
    retrievers = ['colbert', 'dpr']
    rerankers = ['lit5dist-LiT5-Distill-base-v2', 'lit5dist-LiT5-Distill-large-v2', 'lit5dist-LiT5-Distill-xl-v2',
                 'monot5-monot5-3b-msmarco-10k', 'monot5-monot5-base-msmarco-10k', 'monot5-monot5-large-msmarco-10k',
                 'rankgpt-Llama-3.1-8B-Instruct', 'rankgpt-Qwen2.5-7B', 'rankt5-rankt5-3b', 'rankt5-rankt5-base',
                 'rankt5-rankt5-large',
                 'upr-gpt2-large', 'upr-t0-3b', 'upr-t5-large', 'upr-t5-small', ]
    for retriever in tqdm(retrievers):
        merge(retriever, None)

    for retriever in tqdm(retrievers):
        for reranker in rerankers:
            merge(retriever, reranker)


if __name__ == '__main__':
    main()