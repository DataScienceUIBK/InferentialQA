import sys
sys.path.append('../../../')

import json
import argparse
import os.path

from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.models.reranking import Reranking

def get_docs(retriever):
    docs = []
    with open(f'../retrievers/{retriever}_msmarco_output_100.json', 'r') as f:
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


def to_output(docs, retriever, file_name):
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


def main(retriever, reranker_method, reranker_model):
    if "/" in reranker_model:
        file_name = f'./{retriever}/{reranker_method}-{reranker_model.split("/")[1]}.json'
    else:
        file_name = f'./{retriever}/{reranker_method}-{reranker_model}.json'

    if os.path.exists(file_name):
        return

    reranker = Reranking(method=reranker_method, model_name=reranker_model)

    docs = get_docs(retriever)
    reranker.rank(docs)

    os.makedirs(f'./{retriever}', exist_ok=True)
    to_output(docs, retriever=retriever, file_name=file_name)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Determine the retriever and reranker you want to use")
    argparser.add_argument('--retriever', type=str, required=True)
    argparser.add_argument('--reranker_method', type=str, required=True)
    argparser.add_argument('--reranker_model', type=str, required=True)

    args = argparser.parse_args()

    _retriever = args.retriever
    _reranker_method = args.reranker_method
    _reranker_model = args.reranker_model

    main(_retriever, _reranker_method, _reranker_model)
