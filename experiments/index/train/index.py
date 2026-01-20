import os
import json
import random as rnd

rnd.seed(42)

from tqdm import tqdm


def generate_contexts(q):
    q_id = q['id']
    question = q['question']
    hints = [hint['hint'] for hint in q['hints']]
    answers = q['answers']

    contexts = []
    for subset in q['subsets']:
        for perm in q['subsets'][subset]:
            context_id = f'{q_id}_{subset}_{perm}'

            hints_lst = [hints[int(idx)] for idx in perm]
            context_text = ' '.join(hints_lst).strip()

            context_label = 1
            _generated_answers = [q['subsets'][subset][perm][_llm] for _llm in
                                  ["gemma-3-1b", "qwen-3-4b", "llama-31-8b"]]

            correct_answers = [int(_ga in answers) for _ga in _generated_answers]
            _score = correct_answers[0] + correct_answers[1] + correct_answers[2]
            if _score > 0:
                context_label = 2

            rag_answers = {_llm: q['subsets'][subset][perm][_llm] for _llm in
                           ["llama-32-1b", "gemma-3-4b", "qwen-3-8b"]}
            dict_item = {'id': context_id, 'text': context_text, 'rag_answers': rag_answers, 'label': context_label}
            contexts.append(dict_item)

    return {'question': question, 'answers': answers, 'contexts': contexts}


def main():
    with open('../../dataset/train.json', 'r') as f:
        dataset = json.load(f)

    retriever_jsonl = []
    index_jsonl = []

    for q in tqdm(dataset, desc='Generating files'):
        retriever_item = generate_contexts(q)
        retriever_jsonl.append(retriever_item)
        for context in retriever_item['contexts']:
            context_id = context['id']
            context_text = context['text']
            index_item = {'id': context_id, 'contents': context_text, 'title': q['id']}
            index_jsonl.append(index_item)

    with open("index.jsonl", "w") as f:
        for record in tqdm(index_jsonl, desc='Writing index file'):
            json_line = json.dumps(record)
            f.write(json_line + "\n")

    with open("retriever.jsonl", "w") as f:
        for record in tqdm(retriever_jsonl, desc='Writing retriever file'):
            json_line = json.dumps(record)
            f.write(json_line + "\n")

    if os.path.exists("../train/index.jsonl"):
        with open('../train/index.jsonl', 'r') as f:
            train_index_jsonl = f.readlines()

    if os.path.exists("../test/index.jsonl"):
        with open('../test/index.jsonl', 'r') as f:
            test_retriever_jsonl = f.readlines()

    if os.path.exists("../dev/index.jsonl"):
        with open('../dev/index.jsonl', 'r') as f:
            dev_retriever_jsonl = f.readlines()

    if os.path.exists("../train/index.jsonl") and os.path.exists("../test/index.jsonl") and os.path.exists("../dev/index.jsonl"):
        with open('../corpus.jsonl', 'w') as f:
            corpus = train_index_jsonl + test_retriever_jsonl + dev_retriever_jsonl
            rnd.shuffle(corpus)
            f.writelines(corpus)

if __name__ == '__main__':
    main()
