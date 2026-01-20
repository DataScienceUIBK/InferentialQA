import json
import argparse
from tqdm import tqdm

def main():
    candidates = dict()
    with open(f'./{base_path}/run.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        q_id, ctx_id, score = line.split('\t')
        if q_id not in candidates:
            candidates[q_id] = []
        candidates[q_id].append({'id': ctx_id, 'score': float(score.strip())})
    candidates = list(candidates.values())

    documents = dict()
    with open('../../index/corpus.jsonl', 'r') as f:
        lines = f.readlines()
    for line in lines:
        line_dict = json.loads(line)
        ctx_id = line_dict['id']
        line_dict['text'] = line_dict['contents']
        del line_dict['contents']
        documents[ctx_id] = line_dict

    results = []
    with open('../../index/test/retriever.jsonl', 'r') as f:
        lines = f.readlines()
    for idx, line in enumerate(tqdm(lines)):
        line_dict = json.loads(line)
        query = dict()
        query['question'] = line_dict['question']
        query['answers'] = line_dict['answers']
        query['ctxs'] = []
        for can in candidates[idx]:
            can_id = can['id']
            score = can['score']
            text = documents[can_id]['text']
            title = documents[can_id]['title']
            has_answer = any([ans.lower().strip() in text.lower().strip() for ans in query['answers']])
            ctx_dict = {'score': score, 'has_answer': has_answer, 'id': can_id, 'text': text, 'title': title}
            query['ctxs'].append(ctx_dict)
        results.append(query)

    with open(f'./{base_path}/finetuned_dpr_{base_path}_output_1000.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_passages",  type=int, required=True)
    args = parser.parse_args()

    num_of_passages = int(args.num_of_passages)
    base_path = f'{num_of_passages}_positive'

    main()
