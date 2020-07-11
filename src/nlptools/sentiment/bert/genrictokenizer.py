"""
    Generic Tokenizer class 

"""
import torch
import random
import logging
from torchtext import data, datasets
from transformers import BertTokenizer, AlbertTokenizer

__version__ = 0.1

"""
    Tokenizer for generating tokens , masks and padding 

    TODO: need to work on t5 
"""


class GenericTokenizer(object):

    def __init__(self, model='bert'):

        if model.lower() == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.max_input_length = self.tokenizer.max_model_input_sizes['bert-base-uncased']
        elif model.lower() == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.max_input_length = self.tokenizer.max_model_input_sizes['albert-base-v2']
        elif model.lower() == 't5':
            pass

        # these fields are required for dataset.Field and for any masked model

        self.init_token = self.tokenizer.cls_token  # classification token
        self.eos_token = self.tokenizer.sep_token  # separation token
        self.pad_token = self.tokenizer.pad_token  # padding token
        self.unk_token = self.tokenizer.unk_token  # unknown token

        # indexices for the above tokens

        self.init_token_idx = self.tokenizer.cls_token_id
        self.eos_token_idx = self.tokenizer.sep_token_id
        self.pad_token_idx = self.tokenizer.pad_token_id
        self.unk_token_idx = self.tokenizer.unk_token_id

    def tokenize(self, sentence):
        logging.info("inside the tokenizer")
        tokens = self.tokenizer.tokenize(sentence)
        # one for start and one for end
        tokens = tokens[:self.max_input_length - 2]
        return tokens
