import os
import re
import glob
import torch
import random
import logging
from t5utils import GenericUtils
from torch.utils.data import Dataset


"""
IMDB Folder structure : aclImdb

test/pos train/neg which consists of txt files with review about it

"""

logger = logging.getLogger(__name__)

CONFIG_FILE = "imdbconfig.json"  # do not modify it


class ImdbDatasetTokens(Dataset):

    """
        Dataset for loading imdb files and converting them into tokens.
        Configuration is read from the imbbconfig file 

        :param type_path: train or test or val 
        :type: string
        :param tokenizer: t5 tokenizer
        :type : tokenizer

    """

    def __init__(self, train_test_val, tokenzier):

        super().__init__()

        self.config = GenericUtils.load_config(
            CONFIG_FILE)

        # data folder for imdb dataset
        data_dir = self.config['data_dir']

        # check if the data exists,else create it
        self.prepare_dataset()

        self.pos_file_path = os.path.join(data_dir, train_test_val, 'pos')
        self.neg_file_path = os.path.join(data_dir, train_test_val, 'neg')

        self.pos_files = glob.glob("%s/*.txt" % self.pos_file_path)
        self.neg_files = glob.glob("%s/*.txt" % self.neg_file_path)

        self.max_len = self.config['max_len']  # dont override unless required
        self.target_len = self.config['target_len']

        self.tokenzier = tokenzier  # tokenzier is here !!

        self.data = []
        self.targets = []

        self.tokeneize(self.pos_files, 'positive')
        self.tokeneize(self.neg_files, 'negative')

    """
        check if dataset is downloaded and extracted if not download from url provided
        in json file and creates train , test and validation sets from the configuration.
        
    """

    def prepare_dataset(self):

        data_dir = self.config['data_dir']

        # check if train directory exists if not download the data
        if os.path.exists(data_dir + "/train/neg/") and os.path.exists(data_dir + "/train/pos/"):
            logger.info(
                "training directory exists so not downloading again..")

        else:
            logger.info(" IMDB dataset is not present , downloading ..... ")
            GenericUtils.download_and_extract(
                self.config['download_url'], self.config['tar_file_name'])

        # check if validation directory exists if not extract/create
        if not (os.path.exists(data_dir + "/val/pos") and os.path.exists(data_dir + "/val/neg/")):
            try:
                os.makedirs(data_dir + "/val/pos")
                os.makedirs(data_dir + "/val/neg/")
            except OSError:
                logger.error("Creation of the validation directory failed")

            # create validation data from training data with fraction from config file
            train_pos_files = glob.glob(data_dir + '/train/pos/*.txt')
            train_neg_files = glob.glob(data_dir + '/train/neg/*.txt')
            train_pos_len = len(train_pos_files)
            train_neg_len = len(train_neg_files)

            assert len(train_pos_len) == 12500 and len(train_neg_len) == 12500

            if train_pos_len != 12500 and train_neg_len != 12500:
                logger.warn("train data set is modified")

            # create random shuffle from training set
            random.shuffle(train_pos_files)
            random.shuffle(train_neg_files)

            val_pos_files = train_pos_files[:self.config['val_split']]
            val_neg_files = train_neg_files[:self.config['val_split']]

            import shutil
            for f in val_pos_files:
                shutil.move(f,  data_dir + '/val/pos')
            for f in val_neg_files:
                shutil.move(f,  data_dir + '/val/neg')

    """
        @override --> len to return input data , in accordance with Dataset

        returns: length of input data array
        rtype: int

    """

    def __len__(self):
        return len(self.data)

    """
        @overriding method so same syntax.

        T5 requies input ids and seqence pairs to encoding them

        :param index : index to iterate through collection
        :type: int
        :return : json with input ids and attention_mask for modelling
        :type : string

    """

    def __getitem__(self, index):

        # 511 ids for reviews + </s>
        source_ids = self.data[index]["input_ids"].squeeze()  # (512,)
        src_mask = self.data[index]["attention_mask"].squeeze()

        # 2 , positive/negative id + </s>
        target_ids = self.targets[index]["input_ids"].squeeze()  # (2,)
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    """
        Read the files, tokenize the inputs and targets and return them.
        T5 needs </s> token to be added at the end to for EOS for inputs and target.

        using T5 tokenizer's batch_encode_plus method, encoding the inputs required for
        T5 model basicaly text -> text format.

        :param riles: files path to read from.
        :type: string
        :param pos_neg: either postive or negative directory to read from.
        :type: string

    """

    def tokeneize(self, files, pos_neg):

        # consolidated regexs for all edge  cases to remove spaces , colons for creating tokens
        REG1 = re.compile(
            "[.;:!\'?,\"()\[\]]")  # pylint: disable=anomalous-backslash-in-string
        REG2 = re.compile(
            "(<br\s*/><br\s*/>)|(\-)|(\/)")  # pylint: disable=anomalous-backslash-in-string

        for _file in files:
            with open(_file, 'r') as f:
                line = f.read()

            line = line.strip()
            line = re.compile(REG1).sub("", line)
            line = re.compile(REG2).sub("", line)
            line = line + " </s>"  # this is the EOF for T5 !! different from bert # max_len -> 512

            target = pos_neg + " </s>"  # even add this as EOS for output

            tokenized_input = self.tokenzier.batch_encode_plus(
                [line],
                max_length=self.max_len,  # max length of the review !
                pad_to_max_length=True,
                return_tensors="pt"  # we love pytorch !!
            )

            tokenized_target = self.tokenzier.batch_encode_plus(
                [target],
                max_length=self.target_len,  # 1 -> one embed_id either negative/postive +  1 -> </s>
                pad_to_max_length=True,
                return_tensors="pt"  # we love pytorch !!
            )

            self.data.append(tokenized_input)
            self.targets.append(tokenized_target)
