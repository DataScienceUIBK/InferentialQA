import sys
sys.path.append('../')
from hf_auto_open import open_json

import json
from tqdm import tqdm
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

import nltk
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


def total_questions(pairs):
    qs = set([pair[0] for pair in pairs])
    return len(qs)

def total_passages(pairs):
    return len([pair[1] for pair in pairs])

def avg_question_length(pairs):
    qs = set([pair[0] for pair in pairs])
    return sum([len(q.split(' ')) for q in qs]) / len(qs)

def avg_passage_length(pairs):
    return sum([len(pair[1].split(' ')) for pair in pairs]) / len(pairs)

def avg_answer_length(pairs):
    anses = []
    [anses.extend(pair[2]) for pair in pairs]
    return sum([len(ans.split(' ')) for ans in anses]) / len(anses)


def query_passage_overlap(pairs):
    overlap = 0
    for pair in pairs:
        question, passage, answers = pair
        question = question.split(' ')
        passage = passage.split(' ')

        question = set([w for w in question if w not in stop_words])
        passage = set([w for w in passage if w not in stop_words])

        intersection = question.intersection(passage)
        union = question.union(passage)
        jaccard = len(intersection) / len(union)
        overlap += jaccard
    return overlap / len(pairs)


def answer_containment(pairs):
    overlap = 0
    for pair in pairs:
        question, passage, answers = pair
        for answer in answers:
            if passage.lower().find(answer.lower()) >= 0:
                overlap += 1
                break
    return overlap / len(pairs)


def semantic_similarity(pairs):
    similarity = 0
    model = SentenceTransformer("all-MiniLM-L6-v2")

    pairs_dict = dict()
    for pair in pairs:
        _q, _p, _ = pair

        if _q not in pairs_dict:
            pairs_dict[_q] = []
        pairs_dict[_q].append(_p)

    for q in tqdm(pairs_dict):
        question = [q]
        passages = pairs_dict[q]

        q_embeddings = model.encode(question)
        p_embeddings = model.encode(passages)

        similarities = model.similarity(q_embeddings, p_embeddings)
        similarity += similarities.sum().item()
    return similarity / len(pairs)


def main():
    pairs = []
    for _set in tqdm(['train', 'dev', 'test']):
        lines = open_json(f'index/{_set}/retriever.jsonl')
        for line in lines:
            line = json.loads(line)
            question = line['question']
            answers = line['answers']
            ctxs = line['contexts']
            for ctx in ctxs:
                pairs.append((question, ctx['text'], answers))
    print('Total Questions:', total_questions(pairs))
    print('Total Passages:', total_passages(pairs))

    print('Average Question Length:', avg_question_length(pairs))
    print('Average Passage Length:', avg_passage_length(pairs))
    print('Average Answer Length:', avg_answer_length(pairs))

    print('Query-Passage overlap:', query_passage_overlap(pairs))
    print('Answer-Containment:', answer_containment(pairs))
    print('Semantic similarity:', semantic_similarity(pairs))


if __name__ == '__main__':
    main()
