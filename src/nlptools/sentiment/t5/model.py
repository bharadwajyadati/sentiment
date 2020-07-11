import torch
import logging
import pytorch_lightning as pl
from t5utils import GenericUtils
from logcallback import CallbackLogger
from torch.utils.data import DataLoader
from transformers import TFT5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup, T5Tokenizer


"""
    T5 training class using lighting module ! thus instead of
    using nn.module , using LightingModule

    for tokenize -- t5 base ,read from config can be changed
    for model , fine tune -- t5 base ,read from config can be changed
 
    most of the code is inspired from
    https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/core/lightning.py

    since LightingModule,need to override few methonds and logging is done using tenrsorboard too .!

    Make sure it runs on GPU !

"""

logger = logging.getLogger(__name__)

CONFIG_FILE = "modelconfig.json"  # do not modify it

"""
    Generic T5 model for both sentiment and emotion detection, most of the things can be
    resused except the output folder to store the trained model

"""


class T5Model(pl.LightningModule):

    """
        Initialize  t5 tokenizier and t5 model from the config file

        :param task_name: different tasks to use T5 , presently sentiment and emotion detection
        :type: string

    """

    def __init__(self, task_name):
        super().__init__()

        self.task_name = task_name  # sentiment or emotion classifier

        try:
            self.config = GenericUtils.load_config(
                CONFIG_FILE)
            model_name = self.config['sentiment']['model_name']

            # TODO: save this models for future use, or change to s3
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = TFT5ForConditionalGeneration.from_pretrained(
                model_name)

        except Exception as exp:
            logger.error("expection" + exp)

    """
        Lighting logger

    """

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    """ 
        Fine turning T5 as is, we are not adding any layers at end !! coz t5 works !

    """

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,  # Indices of input sequence tokens in the vocabulary.
            attention_mask=attention_mask,  # Mask to perform attention on padding token indices
            # T5 uses the pad_token_id as the starting token for decoder_input_ids generation.
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,)  # since its a sequence classification task !

    """
        T5 is encoder and decoder network 

        encoder needs:
                      source_ids
                      source_masks

        Decoder needs:
                      target_mask   

        used in forward pass to calculate loss per step or batch , hence the name !         

    """

    def _step(self, batch):
        # Labels for computing the sequence classification/regression loss.
        lm_labels = batch["target_ids"]
        # Change pad_token_id to -100 , All labels set to -100 are ignored (masked)
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )
        # get loss from output
        loss = outputs[0]

        return loss

    """
        Runs forward pass --> calculate loss --> return loss
    """

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    """ 
        similar to forward pass for training , if this needs to be used in kaggle , 
        then include jarccard_score here !

    """

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    """
        Linear warmup and decay optimizer and schedular
    """

    def configure_optimizers(self):

        model = self.model

        lr = self.config['learning_rate']
        eps = self.config['adam_epsilon']
        weight_decay = self.config['weight_decay']

        # do not apply weight decay to "bias" and "LayerNorm.weight" parameters
        no_decay = ["bias", "LayerNorm.weight"]
        # similarly for grouped parameters which wont require weight decay
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # Adam with weight decay
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=lr, eps=eps)
        self.opt = optimizer
        return [optimizer]

    """
        Adjust weights for gradients and lr scheduler
    """

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        # in case of xla --> GCP or colab
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)  # pylint: disable=undefined-variable
        else:
            optimizer.step()  # GPU , adjust weights
        # set them to zero
        optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(
            self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):

        train_batch_size = self.config['train_batch_size']

        train_dataset = self.get_dataset(train_test_val="train")

        dataloader = DataLoader(train_dataset, batch_size=train_batch_size,
                                drop_last=True, shuffle=True, num_workers=self.config['num_workers'])
        t_total = (
            (len(dataloader.dataset) //
             (train_batch_size * max(1, self.config['n_gpu'])))
            // self.config['gradient_accumulation_steps']
            * float(self.config['num_train_epochs'])
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.config['warmup_steps'], num_training_steps=t_total)
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = self.get_dataset(train_test_val="val")
        return DataLoader(val_dataset, batch_size=self.config['eval_batch_size'], num_workers=self.config['num_workers'])

    def get_dataset(self, train_test_val):
        if self.task_name == "sentiment":
            from imdbdataset import ImdbDatasetTokens
            return ImdbDatasetTokens(train_test_val, self.tokenizer)
        elif self.task_name == "emotion_classifer":
            pass
