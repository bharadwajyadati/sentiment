import os
import json
import torch
import random
import logging
import numpy as np
from model import T5Model
from tqdm.auto import tqdm
from sklearn import metrics
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from imdbdataset import ImdbDatasetTokens
from logcallback import CallbackLogger
from transformers import T5Tokenizer
from t5utils import GenericUtils

CONFIG_FILE = "modelconfig.json"

logger = logging.getLogger(__name__)

"""
    Generic class to train any task

    Usage:  
    
        Train:
                sentiment_task = TStrainer("sentiment")
                sentiment_task.fit()
            
        Evaluate:
                sentiment_task.evaluate()


    #NOTE: Training takes more time ,even on a gpu , so be patient ! :)

        
"""


class T5trainer(object):

    """ 

        Generic trainer for both the tasks, output dir and task_name are diff for both

        :param task_type: sentiment , emotional
        :type: String


    """

    def __init__(self, task_type):

        self.task_type = task_type
        self.seed = 42  # !

        try:
            self.config = GenericUtils.load_config(CONFIG_FILE)
        except:
            logging.error("error loading config file")

        self.tokenizer = T5Tokenizer.from_pretrained(
            self.config['tokenizer'])

    """
        Set all the seeds to make results replicable
        
    """

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    """
        Training is done here , make sure this happens on gpu ! if possible use multi-gpu.

        Once training is done , model is stored in output_dir from config along with checkpoints

        #TODO: need to exhance to use multi-gpu  
        #TODO: work on providing versioning

    """

    def train(self):

        config = self.config

        chk_dir = config[self.task_type]["chk_dir"]
        output_dir = config[self.task_type]["ouput_dir"]
        if not os.path.exists(chk_dir) and not os.path.exists(output_dir):
            os.makedirs(chk_dir)
            os.makedirs(output_dir)
        if torch.cuda.is_available():  # need to handle multi gpu case
            n_gpu = 1

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=chk_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5)

        train_params = dict(
            accumulate_grad_batches=config["gradient_accumulation_steps"],
            gpus=n_gpu,
            max_epochs=config["num_train_epochs"],
            early_stop_callback=False,
            precision=16 if config["fp_16"] else 32,
            amp_level=config["opt_level"],
            gradient_clip_val=config["max_grad_norm"],
            checkpoint_callback=checkpoint_callback,
            callbacks=[CallbackLogger(chk_dir)],)

        # init the model
        model = T5Model(self.task_type)

        trainer = pl.Trainer(**train_params)

        logger.info(" Started Training .....")

        trainer.fit(model)  # will take time

        logger.info(" Training done.. saving the model..")

        # make sure this exists !, provide versioning !
        model.model.save_pretrained(output_dir)

        self.model = model

    """
        After traning is done , to test perfomance on test set , use this.Generates accuracy report

    """

    def evaluate(self):
        train_test_val = "test"
        dataset = ImdbDatasetTokens(train_test_val, self.tokenizer)
        loader = DataLoader(
            dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'])

        self.model.model.eval()
        outputs = []
        targets = []
        for batch in tqdm(loader):
            if torch.cuda.is_available():
                outs = self.model.model.generate(input_ids=batch['source_ids'].cuda(),
                                                 attention_mask=batch['source_mask'].cuda(
                ), max_length=self.config[self.task_type]['target_len'])
            else:
                outs = self.model.model.generate(input_ids=batch['source_ids'],
                                                 attention_mask=batch['source_mask'], max_length=2)
            decoded_output = [self.tokenizer.decode(ids) for ids in outs]
            output = [self.tokenizer.decode(ids)
                      for ids in batch["target_ids"]]

        outputs.extend(decoded_output)
        targets.extend(output)
        acc = metrics.accuracy_score(targets, outputs)
        logger.info("Accuracy score is " + acc)
        report = metrics.classification_report(targets, outputs)
        print(report)
        logger.info(report)
