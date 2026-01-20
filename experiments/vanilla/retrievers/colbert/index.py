import shutil
import os
import json
from tqdm import tqdm
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.data import Queries
from colbert import Searcher


def retrieve():
    with Run().context(RunConfig(nranks=1, experiment="msmarco", root="./index/corpus_dir")):
        config = ColBERTConfig(
            nbits=2, doc_maxlen=256, query_maxlen=64, bsize=128
        )
        indexer = Indexer(checkpoint='colbert-ir/colbertv2.0', config=config)
        indexer.index(name="latent_ir", collection="./corpus_dir/collection_test.tsv")

    with Run().context(RunConfig(nranks=1, experiment="msmarco", root="./index/corpus_dir")):
        config = ColBERTConfig()
        searcher = Searcher(index="latent_ir", config=config)
        queries = Queries("./dev_dir/queries_test.tsv")
        ranking = searcher.search_all(queries, k=1000)
        ranking.save("latent_ir.ranking.tsv")

    shutil.move(f'./index/corpus_dir/msmarco/indexes/latent_ir',
                f'./index/corpus_dir')

    checkpoint_dir = f'./index/corpus_dir'
    for _ in range(5):
        checkpoint_dir = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
    shutil.move(f'{checkpoint_dir}/latent_ir.ranking.tsv', f'./index/run.txt')
    shutil.rmtree(f'./index/corpus_dir/msmarco')


def to_json():
    candidates = dict()
    base_path = f'index'

    with open(f'./corpus_dir/id_map_test.json', 'r') as f:
        corpus_id_map = json.load(f)
    with open(f'./dev_dir/id_map_test.json', 'r') as f:
        query_id_map = json.load(f)

    with open(f'./{base_path}/run.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        q_id, ctx_id, _, score = line.split('\t')
        q_id = query_id_map[q_id]
        ctx_id = corpus_id_map[ctx_id]
        if q_id not in candidates:
            candidates[q_id] = []
        candidates[q_id].append({'id': ctx_id, 'score': float(score.strip())})
    candidates = list(candidates.values())

    documents = dict()
    with open('../../../index/corpus.jsonl', 'r') as f:
        lines = f.readlines()
    for line in lines:
        line_dict = json.loads(line)
        ctx_id = line_dict['id']
        line_dict['text'] = line_dict['contents']
        del line_dict['contents']
        documents[ctx_id] = line_dict

    results = []
    with open('../../../index/test/retriever.jsonl', 'r') as f:
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

    with open(f'./colbert_output_1000.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    retrieve()
    to_json()
