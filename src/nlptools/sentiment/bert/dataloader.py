import torch
import random
import logging
import pandas as pd
from transformers import BertTokenizer, AlbertTokenizer
from torchtext import data, datasets

"""
    DataLoader for IMDB , SST or custom datasets

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


class DataLoader(object):

    def __init__(self, model="bert"):
        self.max_input_length = 128
        # check if gpu is present for faster training
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # to get same output
        self.random_seed = 1111

        # model parameters
        self.batch_size = 128  # tryout with 256

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

    def load_dataset(self):
        pass

    def iterator(self):
        pass

    def tokenize(self, sentence):
        logging.info("inside the tokenizer")
        tokens = self.tokenizer.tokenize(sentence)
        # one for start and one for end
        tokens = tokens[:self.max_input_length - 2]
        return tokens


class IMDB_SSTDataLoader(DataLoader):

    def __init__(self, model="bert"):
        super().__init__(model)

    """
        Uses text.Field and text.Label

    """

    def text_label(self):

        TEXT = data.Field(
            batch_first=True,
            use_vocab=False,
            tokenize=self.tokenize,  # tokenize method
            preprocessing=self.tokenizer.convert_tokens_to_ids,  # for tokenizng
            init_token=self.init_token_idx,
            eos_token=self.eos_token_idx,
            pad_token=self.pad_token_idx,
            unk_token=self.unk_token_idx
        )

        LABEL = data.LabelField(dtype=torch.float)
        return TEXT, LABEL

    """
        remove two tokens along with one for start and one for end
    """

    def tokenize(self, sentence):
        logging.info("inside the tokenizer")
        tokens = self.tokenizer.tokenize(sentence)
        # one for start and one for end
        tokens = tokens[:self.max_input_length - 2]
        return tokens

    """
        since its sentiment analysis , use either SST or IMDB or custom

        TODO: try dataloader and read FinBert data set and train nit
    """

    def load_dataset(self, dataset="imdb"):

        TEXT, LABEL = self.text_label()

        logging.info("inside load dataset")

        train_data, test_data, validation_data = "", "", ""

        if dataset.lower() == 'imdb':
            train_data, test_data = datasets.IMDB.splits(
                text_field=TEXT, label_field=LABEL)

            print(train_data)
            train_data, validation_data = train_data.split(
                random_state=random.seed(self.random_seed))
        elif dataset.lower() == "sst":
            train_data, test_data = datasets.SST.splits(
                text_field=TEXT, label_field=LABEL)

            train_data, validation_data = train_data.split(
                random_state=random.seed(self.random_seed))

        LABEL.build_vocab(train_data)  # build vocab for labels

        return train_data, validation_data, test_data

    """
        iterator for data
    """

    def iterator(self):

        train_data, validation_data, test_data = self.load_dataset()

        train_iterator, validation_iterator, test_iterator = data.BucketIterator.splits((train_data, validation_data, test_data),
                                                                                        batch_size=self.batch_size, device=self.device)

        return train_iterator, validation_iterator, test_iterator


"""
    Crude , needs to be updated

"""


class FinBerDataLoader(DataLoader):

    def __init__(self, file, tokenizer, seperator="@"):
        super().__init__(tokenizer)
        self.seperator = seperator
        self.file = file

    def load_dataset(self):
        dataset = {}
        dataset['sentence'] = []
        dataset['sentiment'] = []
        with open(self.file, encoding="ISO-8859-1") as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.split(self.seperator)
            print(line)
            dataset['sentence'].append(line[0].strip())
            dataset['sentiment'].append(line[1].strip())
        return pd.DataFrame.from_dict(dataset)

    def iterator(self):
        pass

