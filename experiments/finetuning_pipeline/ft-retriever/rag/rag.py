import json
import argparse
from by_server import By_Server


def rag(model, retriever, reranker):
    if reranker is None:
        with open(f'./cleaned/retrievers/{retriever}_output_1000.json', 'r') as f:
            json_data = json.load(f)
    else:
        with open(f'./cleaned/rerankers/{retriever}/{reranker}.json', 'r') as f:
            json_data = json.load(f)

    # rag_input = []
    # for q in json_data:
    #     q_id = q['id']
    #     ctx_id = q['ctx']['id']
    #     ctx_id_part = '_'.join(ctx_id.split('_')[:3])
    #     if q_id != ctx_id_part:
    #         rag_input.append(q)
    # qa = By_Server(model, retriever, reranker, rag_input)
    qa = By_Server(model, retriever, reranker, json_data)
    qa.qa()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Results of RAG pipeline...")
    parser.add_argument('--model', type=str, help='Model to generate answers')
    parser.add_argument('--retriever', type=str, help='The retriever to use')
    parser.add_argument('--reranker', type=str, help='The reranker to use')
    args = parser.parse_args()

    model = args.model
    retriever = args.retriever
    reranker = args.reranker

    rag(model, retriever, reranker)
