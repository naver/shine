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
from fileios import *


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


class HrchyPrompter:
    def __init__(self, dataset_name, num_sub=10, num_super=0, method=None):
        self.dataset_name = dataset_name
        self.num_sub = num_sub
        self.num_super = num_super
        self.method = method

    def _query_inat(self, node_name, context=''):
        prompt = f"Generate a list of {self.num_sub} {context[0]} of the following {context[1]} and output the list separated by '&' (without numbers): {node_name}"
        return prompt

    def _query_fsod(self, node_name, context=''):
        child_prompt = f"Generate a list of {self.num_sub} {context[0]} of the following {context[1]} and output the list separated by '&' (without numbers): {node_name}"
        parent_prompt = f"Generate a list of {self.num_super} super-categories that the following {context[1]} belongs to and output the list separated by '&' (without numbers): {node_name}"
        return child_prompt, parent_prompt

    def _query_others(self, node_name, context=''):
        child_prompt = f"Generate a list of {self.num_sub} {context[0]} of the following {context[1]} and output the list separated by '&' (without numbers): {node_name}"
        parent_prompt = f"Generate a list of {self.num_super} super-categories that the following {context[1]} belongs to and output the list separated by '&' (without numbers): {node_name}"
        return child_prompt, parent_prompt

    def embed(self, node_name, context=''):
        if self.dataset_name == 'inat':
            return self._query_inat(node_name, context)
        elif self.dataset_name == 'fsod':
            return self._query_fsod(node_name, context)
        elif self.dataset_name in ['oid_lvis', 'lvis', 'oid', 'coco']:
            return self._query_others(node_name, context)
        else:
            raise NameError(f"{self.dataset_name} has NO prompt template")

