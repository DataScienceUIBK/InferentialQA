import json
import argparse
import os.path

from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.models.reranking import Reranking


def get_docs(reranker_model):
    docs = []
    with open(f'./{reranker_model}/test_data.json', 'r') as f:
        retriever_output = json.load(f)
    for q in retriever_output:
        doc = Document(
            question=Question(question=q['question']),
            answers=Answer(q['answers']),
            contexts=[Context(text=ctx['text'], id=ctx['id'], score=ctx['score'], has_answer=ctx['has_answer'],
                              title=ctx['title']) for ctx in q['ctxs']]
        )
        docs.append(doc)
    return docs


def to_output(docs, file_name):
    json_dict = []
    for doc in docs:
        res_dict = dict()
        res_dict['question'] = doc.question.question
        res_dict['answers'] = doc.answers.answers
        res_dict['ctxs'] = []
        for context in doc.contexts:
            ctx = dict()
            ctx['score'] = context.score
            ctx['has_answer'] = context.has_answer
            ctx['id'] = context.id
            ctx['text'] = context.text
            ctx['title'] = context.title
            res_dict['ctxs'].append(ctx)
        res_dict['reranked_ctxs'] = []
        for context in doc.reorder_contexts:
            ctx = dict()
            ctx['score'] = context.score
            ctx['has_answer'] = context.has_answer
            ctx['id'] = context.id
            ctx['text'] = context.text
            ctx['title'] = context.title
            res_dict['reranked_ctxs'].append(ctx)
        json_dict.append(res_dict)
    with open(file_name, 'w') as f:
        json.dump(json_dict, f, indent=4)


def main(reranker_method, reranker_model):
    file_name = f'./{reranker_model}/finetuned_{reranker_method}_{reranker_model}_output.json'

    if os.path.exists(file_name):
        return

    reranker = Reranking(method=reranker_method, model_name=f'./{reranker_model}/checkpoints/checkpoint_final')

    docs = get_docs(reranker_model)
    reranker.rank(docs)

    to_output(docs, file_name=file_name)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Determine the retriever and reranker you want to use")
    argparser.add_argument('--reranker_model', type=str, required=True)

    args = argparser.parse_args()

    _reranker_method = 'monot5'
    _reranker_model = args.reranker_model

    main(_reranker_method, _reranker_model)
