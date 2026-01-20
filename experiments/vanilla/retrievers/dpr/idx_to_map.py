import json

def main():
    idx_to_map = dict()
    with open('../../../index/corpus.jsonl', 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]
    for idx, line in enumerate(lines):
        idx_to_map[idx] = line['id']

    with open('./dpr_output_1000.json', 'r') as f:
        output = json.load(f)

    for q in output:
        for ctx in q['ctxs']:
            ctx['id'] = idx_to_map[int(ctx['id'])]

    with open('./dpr_output_1000.json', 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == '__main__':
    main()