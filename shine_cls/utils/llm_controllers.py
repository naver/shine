import argparse
import json
import torch
import numpy as np
import itertools
from nltk.corpus import wordnet
import sys
import clip
import os
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import re
from copy import deepcopy
import ast


def clean_name(name):
    name = name.replace('<', '')
    name = name.replace('>', '')
    name = name.replace("'''", '')
    name = name.strip()
    return name


def dump_json(filename: str, in_data):
    if not filename.endswith('.json'):
        filename += '.json'

    with open(filename, 'w') as fbj:
        if isinstance(in_data, dict):
            json.dump(in_data, fbj, indent=4)
        elif isinstance(in_data, list):
            json.dump(in_data, fbj)
        else:
            raise TypeError(f"in_data has wrong data type {type(in_data)}")


def load_json(filename: str):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)


def prepare_chatgpt_message(main_prompt):
    messages = [{"role": "user", "content": main_prompt}]
    return messages


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_chatgpt(chatgpt_messages, temperature=0.7, max_tokens=40, model="gpt-3.5-turbo"):
    if max_tokens > 0:
        response = openai.ChatCompletion.create(
            model=model,
            messages=chatgpt_messages,
            temperature=temperature,
            max_tokens=max_tokens)
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=chatgpt_messages,
            temperature=temperature)

    reply = response.choices[0].message["content"]
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens


def trim_question(question):
    question = question.split('Question: ')[-1].replace('\n', ' ').strip()
    if 'Answer:' in question:  # Some models make up an answer after asking. remove it
        q, a = question.split('Answer:')[:2]
        if len(q) == 0:  # some not so clever models will put the question after 'Answer:'.
            question = a.strip()
        else:
            question = q.strip()
    return question


class LLMBot:
    def __init__(self, model_tag, max_chat_token=-1):
        self.model_tag = model_tag
        self.model_name = "ChatGPT"
        self.max_chat_token = max_chat_token
        self.total_tokens = 0

    def reset(self):
        self.total_tokens = 0

    def get_used_tokens(self):
        return self.total_tokens

    def get_name(self):
        return self.model_name

    def __call_llm(self, main_prompt, temperature, max_token):
        total_prompt = prepare_chatgpt_message(main_prompt)
        reply, n_tokens = call_chatgpt(total_prompt, temperature=temperature,
                                       model=self.model_tag, max_tokens=max_token)
        return reply, total_prompt, n_tokens

    def infer(self, main_prompt, temperature=0.7):
        reply, _, n_tokens = self.__call_llm(main_prompt, temperature, max_token=self.max_chat_token)
        reply = reply.strip()
        self.total_tokens += n_tokens
        return reply


class Prompter:
    def __init__(self, dataset_name, method='100isa'):
        self.dataset_name = dataset_name
        self.method = method

    def _embed_inat_prompt(self, leaf_name, in_sentence):
        prompt = \
f"""\
I have one sentence delimited by <> that describe a {leaf_name} object in a photo along its semantic \
hierarchy knowledge (Species, Genus, Family, Order, Class, Phylum), following a is-a relationship.

But, This single sentence description is not popular, intuitive, and informative. These descriptions \
use very academic Latin names. These Latin names are not very common on the Internet.

I first give you some exemples delimited by triple backticks of the single sentence I will give you.

Your task is to perform the following actions:
1 - Understand the sentence delimited by <> I give to you and its semantic hierarchy knowledge that follows a is-a \
relationship.
2 - Rephrase the sentence delimited by <> in as different ways (e.g., more colloquial name) as possible 50 times \
to make it more common, specific, intuitive, and informative. But be careful not to change the original meaning of \
the sentence. Each rephrased sentence must include all the hierarchical information (Species, Genus, Family, Order, \
Class, Phylum). Be creative! Use the most common, simple and straightforward way to rephrase the sentence. 
3 - Output the 50 rephrased sentences using Python list format: \
['rephrased sentence 1', 'rephrased sentence 2', ..., 'rephrased sentence 50']

Example of the sentence:
'''a Storeria occipitomaculata, which is a Storeria, which is a Colubridae, which is a Squamata, which is a Reptilia, which is a Chordata'''
'''a Tantilla gracilis, which is a Tantilla, which is a Colubridae, which is a Squamata, which is a Reptilia, which is a Chordata'''
'''a Tarentola mauritanica, which is a Tarentola, which is a Phyllodactylidae, which is a Squamata, which is a Reptilia, which is a Chordata'''
'''a Terrapene carolina, which is a Terrapene, which is a Emydidae, which is a Testudines, which is a Reptilia, which is a Chordata'''
'''a Terrapene ornata, which is a Terrapene, which is a Emydidae, which is a Testudines, which is a Reptilia, which is a Chordata'''
'''a Thamnophis cyrtopsis, which is a Thamnophis, which is a Colubridae, which is a Squamata, which is a Reptilia, which is a Chordata'''
'''a Thamnophis elegans, which is a Thamnophis, which is a Colubridae, which is a Squamata, which is a Reptilia, which is a Chordata'''
'''a Thamnophis hammondii, which is a Thamnophis, which is a Colubridae, which is a Squamata, which is a Reptilia, which is a Chordata'''
'''a Thamnophis marcianus, which is a Thamnophis, which is a Colubridae, which is a Squamata, which is a Reptilia, which is a Chordata'''
'''a Thamnophis proximus, which is a Thamnophis, which is a Colubridae, which is a Squamata, which is a Reptilia, which is a Chordata'''
'''a Thamnophis sirtalis, which is a Thamnophis, which is a Colubridae, which is a Squamata, which is a Reptilia, which is a Chordata'''
'''a Trachemys scripta, which is a Trachemys, which is a Emydidae, which is a Testudines, which is a Reptilia, which is a Chordata'''
'''a Urosaurus ornatus, which is a Urosaurus, which is a Phrynosomatidae, which is a Squamata, which is a Reptilia, which is a Chordata'''
'''a Uta stansburiana, which is a Uta, which is a Phrynosomatidae, which is a Squamata, which is a Reptilia, which is a Chordata'''
'''a Zootoca vivipara, which is a Zootoca, which is a Lacertidae, which is a Squamata, which is a Reptilia, which is a Chordata'''

Input Sentence: <{in_sentence}>
Output:
"""
        return prompt

    def _fsod_prompt_100describe(self, leaf_name, in_sentence):
        # l2_name = in_sentence.split(", which is a ")[-2].strip()
        # l1_name = in_sentence.split(", which is a ")[-1].strip()

        prompt = \
f"""\
I have one sentence delimited by <> that describe a {leaf_name} object in a photo along its semantic \
hierarchy information, following a is-a relationship.

But, This single sentence description is not popular, intuitive, and informative enough to make people identify \
the {leaf_name} from a photo.

Next, your task is to perform the following actions:
1 - Think about what are the useful features for distinguishing the object delimited by <> in a photo.
2 - Understand the sentence delimited by <> I give to you and its semantic hierarchy information contained in \
the sentence, which follows a is-a relationship along a three-level hierarchy .
3 - Describe the visual features of the object delimited by <> 50 times in as different ways as possible. Each description should be \
intuitive and informative. The goal is to make people easier to distinguish this object from the description.
4 - Output the 50 description sentences using Python list format:
['sentence 1', 'sentence 2', ..., 'sentence 50']

Input Object: <{leaf_name}>
Input Sentence: <{in_sentence}>
Output Descriptions:
"""
        return prompt

    def _fsod_prompt_100isa(self, leaf_name, in_sentence):
        l2_name = in_sentence.split(", which is a ")[-2].strip()
        l1_name = in_sentence.split(", which is a ")[-1].strip()

        prompt = \
f"""\
I have one sentence delimited by <> that describe a {leaf_name} object in a photo along its semantic \
hierarchy information, following a is-a relationship.

But, This single sentence description is not popular, intuitive, and informative enough to make people identify \
the {leaf_name} from a photo. Moreover, this sentence description might not be completely accurate to encode the \
hierarchy information. For example, the sentence '''a double hung window, which is a building, which is a infrastructure''' \
is not accurate. The corrected sentence description will be '''A double hung window is an element of a building, which is a part of infrastructure.'''

I first give you some examples delimited by triple backticks about the single sentence I will give you.

Next, your task is to perform the following actions:
1 - Understand the sentence delimited by <> I give to you and its semantic hierarchy information contained in \
the sentence, which follows a is-a relationship along a three-level hierarchy .
2 - Correct and rephrase the sentence delimited by <> 50 times in as different ways (e.g., more common) as possible \
to make it more accurate, specific, intuitive, and informative. But be careful not to change the original meaning \ 
of the sentence and the semantic hierarchy information contained in the sentence. Each rephrased sentence must \
include all the hierarchical information. Be creative! Use the most accurate, simple and straightforward way to \
rephrase the sentence. 
3 - Output the 50 rephrased sentences using Python list format:
['rephrased sentence 1', 'rephrased sentence 2', ..., 'rephrased sentence 50']

Ten examples of the sentence:
Sentence: '''a beer, which is a drink, which is a liquid'''
Hierarchy information contained in this example sentence: beer --> drink --> liquid

Sentence: '''a fireplace, which is a facility, which is a infrastructure'''
Hierarchy information contained in this example sentence:  fireplace --> facility --> infrastructure

Sentence: '''a musical keyboard, which is a musical instrument, which is a instrument'''
Hierarchy information contained in this example sentence: musical keyboard --> musical instrument --> instrument

Sentence: '''a billboard, which is a facility, which is a infrastructure'''
Hierarchy information contained in this example sentence: billboard --> facility --> infrastructure

Sentence: '''a double hung window, which is a building, which is a infrastructure'''
Hierarchy information contained in this example sentence: double hung window --> building --> infrastructure

Sentence: '''a light switch, which is a electronic product, which is a equipment'''
Hierarchy information contained in this example sentence: light switch --> electronic product --> equipment

Sentence: '''a jet ski, which is a watercraft, which is a vehicle'''
Hierarchy information contained in this example sentence: jet ski --> watercraft --> vehicle

Sentence: '''a shirt button, which is a clothing, which is a wearable item'''
Hierarchy information contained in this example sentence: shirt button --> clothing --> wearable item

Sentence: '''a blue poppy, which is a flower, which is a plant'''
Hierarchy information contained in this example sentence: blue poppy --> flower --> plant

Sentence: '''a cornbread, which is a baked goods, which is a food'''
Hierarchy information contained in this example sentence: cornbread --> baked goods --> food

Sentence: '''  a tuning fork, which is a musical instrument, which is a instrument'''
Hierarchy information contained in this example sentence: tuning fork --> musical instrument --> instrument


Input Sentence: <{in_sentence}>
Hierarchy information contained in the input sentence (from fine-grained to coarse grained): \
{leaf_name} --> {l2_name} --> {l1_name}

Output Sentences:
"""
        return prompt

    def _embed_fsod_prompt(self, leaf_name, in_sentence):
        if self.method == '100isa':
            return self._fsod_prompt_100isa(leaf_name, in_sentence)
        elif self.method == '100describe':
            return self._fsod_prompt_100describe(leaf_name, in_sentence)
        else:
            raise NameError(f"{self.method} not supported")

    def embed(self, leaf_name, in_sentence):
        if self.dataset_name == 'inat':
            return self._embed_inat_prompt(leaf_name, in_sentence)
        elif self.dataset_name == 'fsod':
            return self._embed_fsod_prompt(leaf_name, in_sentence)
        else:
            raise NameError(f"{self.dataset_name} has NO prompt template")

    def _search_children_fsod(self, node_name, node_sentence=None, children_sentences=None):
        if children_sentences is None and node_sentence is not None:
            prompt = \
f"""\
Please perform the following tasks:
1 - Generate 50 sub-categories of '''{node_name}'''.
2 - Use the structure like the example sentence delimited by triple backticks to construct 50 sentences incorporating \
these 50 sub-categories of '''{node_name}'''.
3 - Output the 50 constructed sentences into Python list format like: ['sentence 1', 'sentence 2', ..., 'sentence 50']

Example Sentence: '''{node_sentence}'''

Output Sentences:
"""
#             prompt = \
# f"""
# Please list 50 sub-categories related to a '''{node_name}'''. Afterwards, using the sentence template provided \
# below, construct 50 sentences incorporating these sub-categories. Output the sentences in a Python list format.
#
# Template Sentence: '''{node_sentence}'''
#
# Output Sentences:
# """
            return prompt
        else:
            prompt1 = \
f"""\
I have some sentences delimited by triple backticks. These sentences describe '''{node_name}''' objects following \
a is-a hierarchical relationship.

Your task is to perform the following actions:
1 - Understand the input sentences delimited by triple backticks. 
2 - Generate 50 sentences like the input sentences delimited by triple backticks but do not change the \
word '''{node_name}'''.
3 - Output the 50 generated sentences using Python list format:
['sentence 1', 'sentence 2', ..., 'sentence 50']

Input Sentences:\n
"""
            prompt2 = '\n'.join(["'''{}'''\n".format(sent) for sent in children_sentences])
            prompt3 = "Output Sentences:"
        return prompt1+prompt2+prompt3
        # return prompt2 + "Generate 50 sentences like this into a Python list format:"


    def giveme_children(self, node_name, node_sentence=None, children_sentences=None):
        if self.dataset_name == 'inat':
            pass
        elif self.dataset_name == 'fsod':
            return self._search_children_fsod(node_name, node_sentence, children_sentences)
        else:
            raise NameError(f"{self.dataset_name} has No prompt template")
