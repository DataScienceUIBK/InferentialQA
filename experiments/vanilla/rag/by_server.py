import json
import os
import spacy
from spacy.cli import download

download("en_core_web_sm")

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from termcolor import colored
from tqdm import tqdm


class By_Server:
    def __init__(self, model_url, retriever, reranker, dataset: list):
        self.base_path = f'{retriever}/{reranker}'
        self.retriever = retriever
        self.reranker = reranker
        self.dataset = dataset
        self.steps = 20
        self.device = 'cuda'
        self.spacy_model = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        self.model_url = model_url
        models_map = {'gemma-3-1b-it': 'gemma-3-1b',
                      'llama-3.2-1b-instruct': 'llama-32-1b',
                      'gemma-3-4b-it': 'gemma-3-4b',
                      'qwen3-4b': 'qwen-3-4b',
                      'llama-3.1-8b-instruct': 'llama-31-8b',
                      'qwen3-8b': 'qwen-3-8b'}
        self.model_name = models_map[self.model_url.split('/')[1].lower()]
        self.answers = []
        if os.path.exists(f'./qa_results/{self.base_path}/answers_{self.model_name}.json'):
            with open(f'./qa_results/{self.base_path}/answers_{self.model_name}.json', mode='r', encoding='utf-8') as f:
                self.answers = json.load(f)

    def clean_responses(self, responses: list[str]):
        cleaned_responses = []
        for resp in responses:
            cleaned_resp = resp.strip()
            cleaned_resp = cleaned_resp.split('\n')[-1]
            cleaned_resp = cleaned_resp[1:] if cleaned_resp.startswith('(') else cleaned_resp
            if cleaned_resp == '':
                cleaned_resp = 'NO ANSWER'
            cleaned_responses.append(cleaned_resp)

        is_sentence = []
        docs = self.spacy_model.pipe(cleaned_responses)
        for doc in docs:
            has_valid_sentence = any(
                any(token.dep_ in ("nsubj", "nsubjpass") for token in sent) and
                any(token.pos_ == "VERB" and token.dep_ == "ROOT" for token in sent)
                for sent in doc.sents
            )
            is_sentence.append(has_valid_sentence)

        final_responses = []
        for resp, is_sent in zip(cleaned_responses, is_sentence):
            if is_sent:
                final_cleaned_resp = 'NO ANSWER'
            else:
                final_cleaned_resp = resp
            final_cleaned_resp = 'NO ANSWER' if final_cleaned_resp.lower().find(
                'no answer') >= 0 else final_cleaned_resp
            final_responses.append(final_cleaned_resp)
        return final_responses

    def _prompt(self, batch, model, tokenizer):
        q_ids, prompts = zip(*batch)
        texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for messages in prompts
        ]
        tokenizer.pad_token = tokenizer.eos_token
        # model_inputs = tokenizer(texts, return_tensors="pt", padding=True, pad_to_multiple_of=8).to(self.device)
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True, pad_to_multiple_of=8)
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(
                model_inputs['input_ids'],
                max_new_tokens=32,
                do_sample=False,
                attention_mask=model_inputs['attention_mask'],
                pad_token_id=tokenizer.eos_token_id
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
        ]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        responses = self.clean_responses(responses)
        predict_answers = list(zip(q_ids, responses))
        return predict_answers

    def _to_prompts(self):
        shots = [
            [
                'He was the 44th President of the United States.\nHe served as President from 2009 to 2017.\nHe was the first African-American President of the United States.\nHe was a member of the Democratic Party.\nHe was born on August 4, 1961 in Honolulu, Hawaii.',
                'Who won the Nobel Peace Prize in 2009?', 'Barack Obama'],
            [
                'The capital city of this country is Paris.\nThis country is located in northwestern Europe.\nThis country has a long history and has played a significant role in international affairs.\nThe official language of this country is French.\nThe currency used in this country is the Euro.',
                'Edouard Daladier became Prime Minister of which country in 1933?', 'France'],
            [
                'Its the coldest season of the year.\nIts the season when snow falls in many regions.\nIts the season when many people celebrate Christmas and New Year\'s Eve.\nIts the season when days are shorter and nights are longer.\nIts the season when many animals hibernate.',
                'If you have a \'Mahonia Japonica\', in which season will it be in flower?', 'Winter'],
            [
                'It is a team sport that originated in the United States.\nIt is played with an oval-shaped ball.\nThe objective of the game is to score points by advancing the ball into the opposing team\'s end zone.\nPoints can be scored by carrying the ball across the opponent\'s goal line, throwing it to a teammate in the end zone, or kicking it through the opponent\'s goalposts.\nThe game is divided into four quarters, each lasting 15 minutes.',
                'Which sport is played under the \'Harvard Rules\'?', 'AMERICAN FOOTBALL'],
            [
                'He was born on April 20, 1889 in Braunau am Inn, Austria.\nHe was the leader of the Nazi Party.\nHe became the chancellor of Germany in 1933.\nHe took the title of Führer und Reichskanzler in 1934.\nHe initiated World War II in Europe by invading Poland on September 1, 1939.',
                'Who was made an honorary citizen of Haslach, Austria, in 1938, an honour withdrawn in 2004?',
                'Adolf Hitler']
        ]
        system_prompt = 'You are an assistant that answers questions based on the provided context. You just answer questions with exact answers. You do not use sentences as the response.'
        prompt_cmd = ("Use the context to answer the question under conditions: "
                      '1. Answer should not be sentences. It should be some words.'
                      '2. Do not generate "sorry" or "I cannot ..." sentences, instead, use "NO ANSWER".'
                      '3. Do not generate explanations, reasoning, or full sentences—only provide the exact answer.'
                      '4. If the answer cannot be guessed from the context, respond only with "NO ANSWER".'
                      )

        prompt = "{_PROMPT_CMD}\n\nContext:\n{_SHOT}\n\nQuestion:\n{_QUESTION}\n\nAnswer:\n"
        few_shot_examples_with_context = []
        system_prompt_dict = {"role": "system", "content": system_prompt}
        few_shot_examples_with_context.append(system_prompt_dict)
        for shot in shots:
            user_prompt_dict = {"role": "user",
                                "content": prompt.format(_PROMPT_CMD=prompt_cmd, _SHOT=shot[0], _QUESTION=shot[1])}
            few_shot_examples_with_context.append(user_prompt_dict)
            assistant_prompt_dict = {"role": "assistant", "content": shot[2]}
            few_shot_examples_with_context.append(assistant_prompt_dict)

        prompt = "{_PROMPT_CMD}\n\nQuestion:\n{_QUESTION}\n\nAnswer:\n"
        few_shot_examples_without_context = []
        system_prompt_dict = {"role": "system", "content": system_prompt}
        few_shot_examples_without_context.append(system_prompt_dict)
        for shot in shots:
            user_prompt_dict = {"role": "user",
                                "content": prompt.format(_PROMPT_CMD=prompt_cmd, _QUESTION=shot[1])}
            few_shot_examples_without_context.append(user_prompt_dict)
            assistant_prompt_dict = {"role": "assistant", "content": shot[2]}
            few_shot_examples_without_context.append(assistant_prompt_dict)

        prompts = []
        for rec in self.dataset:
            question = rec['question']
            for ctx in rec['ctxs']:
                context = ctx['text']
                ctx_id = ctx['id']
                question_prompt = "{_PROMPT_CMD}\n\nContext:\n{_SHOT}\n\nQuestion:\n{_QUESTION}\n\nAnswer:\n".format(
                    _PROMPT_CMD=prompt_cmd, _SHOT=context, _QUESTION=question)
                prompt = few_shot_examples_with_context + [{"role": "user", "content": question_prompt}]
                prompts.append((ctx_id, prompt))
        prompts = sorted(prompts, key=lambda x: len(x[1][10]['content']))
        return prompts

    def qa(self):
        os.makedirs(f'./qa_results/{self.base_path}', exist_ok=True)
        prompts = self._to_prompts()
        batch_size = 128

        print(colored(f'Loading {self.model_name} to generate answers for {self.retriever} - {self.reranker}', 'yellow'))
        print()

        qa_model = AutoModelForCausalLM.from_pretrained(
            self.model_url,
            torch_dtype=torch.float32 if self.model_name.find('gemma_3') >= 0 else "auto",
            device_map="auto"
        ).eval()

        qa_model.generation_config.temperature = None
        qa_model.generation_config.top_p = None
        qa_model.generation_config.top_k = None

        tok = AutoTokenizer.from_pretrained(self.model_url, padding_side='left')

        tq = tqdm(total=len(prompts))
        idx = 0
        counter = 0
        while idx < len(prompts):
            if idx < len(self.answers):
                idx += 1
                tq.update(1)
                continue
            while batch_size > 0:
                start_batch = idx
                end_batch = idx + batch_size
                tq.set_description(
                    '{} - {} '.format(idx, len(prompts)))
                batch = prompts[start_batch:end_batch]
                try:
                    predicted_answers = self._prompt(batch, qa_model, tok)
                    tq.update(batch_size)
                    idx = end_batch
                    counter += 1
                    break
                except Exception as ex:
                    batch_size = int(batch_size / 2)
                    tq.set_postfix_str(colored(f'{batch_size * 2}->{batch_size}', 'green'))
            self.answers.extend(predicted_answers)
            if counter % self.steps == 0 or idx >= len(prompts) - 1:
                counter = 0
                with open(f'./qa_results/{self.base_path}/answers_{self.model_name}.json', mode='w',
                          encoding='utf-8') as f:
                    json.dump(self.answers, f, indent=4)
        tq.close()
        print()
