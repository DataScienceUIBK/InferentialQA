import json


def main():
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

    with open('./test.json', 'w') as f:
        json.dump(cleaned_test, f, indent=4)


if __name__ == '__main__':
    main()
