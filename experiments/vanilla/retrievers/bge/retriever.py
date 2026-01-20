import json
from tqdm import tqdm
from rankify.retrievers.bge_retriever import BGERetriever
from rankify.dataset.dataset import Document, Question, Answer


def get_queries():
    queries = []
    with open('../../../index/test/retriever.jsonl', 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        json_dict = json.loads(line)
        query = Document(
            question=Question(question=json_dict['question']),
            answers=Answer(json_dict['answers'])
        )
        queries.append(query)
    return queries


def to_output(results):
    json_dict = []
    for result in results:
        res_dict = dict()
        res_dict['question'] = result.question.question
        res_dict['answers'] = result.answers.answers
        res_dict['ctxs'] = []
        for context in result.contexts:
            ctx = dict()
            ctx['score'] = float(context.score)
            ctx['has_answer'] = context.has_answer
            ctx['id'] = context.id
            ctx['text'] = context.text
            ctx['title'] = context.title
            res_dict['ctxs'].append(ctx)
        json_dict.append(res_dict)
    with open('./bge_output_1000.json', 'w') as f:
        json.dump(json_dict, f, indent=4)


def main():
    retriever = BGERetriever(
        n_docs=1000,
        index_type="wiki",
        index_folder="./index/bge_index_wiki",
        device="cpu"
    )

    queries = get_queries()
    results = retriever.retrieve(queries)
    to_output(results)


if __name__ == "__main__":
    main()
