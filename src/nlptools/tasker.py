"""
    This is like sklearn fit and predict
    Generic api for all the tasks
    default framwork is torch

    https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

"""

import os
import torch
import logging
import allennlp_models.tagging
from sentiment.t5.t5utils import GenericUtils
from allennlp.predictors.predictor import Predictor
from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


logger = logging.getLogger(__name__)

__version__ = 0.1

BERT_MODEL_PATH = "sentiment/bert/sentimentmodel.pt"
T5_MODEL_PATH = "sentiment/t5/t5_base_imdb_sentiment"
T5_MODEL_JSON = "sentiment/t5/modelconf.json"

TASK_FORMAT = {
    "task_name": {
        "task_class": "default"
    }
}


"""
    parent class for all tasks, every other task need to derive from this.!

"""


class Task(object):

    def __init__(self, sentence):
        self.sentence = sentence
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self):
        pass


"""
    Sentiment analysis task , trained with both bert and t5 , but use only t5, cause
    1) t5 is more accurate on imdb dataset -- 94 % compared to bert -92 %
    2) t5 works with phrases even , presently we can use 511 tokens (words) , can be configured and retrained

"""


class SentimentTask(Task):

    def __init__(self, sentence, model="T5"):
        super().__init__(sentence)

        self.model_name = model

        if model == "bert":

            if not os.path.isfile(BERT_MODEL_PATH):
                raise Exception(
                    "trained model is not found , either retrain or download")
            from sentiment.bert.genrictokenizer import GenericTokenizer
            from sentiment.bert.sentimentmodel import BertSentimentModel
            self.tokenizer = GenericTokenizer("bert")
            self.model = BertSentimentModel()
            self.model.load_state_dict(torch.load(
                BERT_MODEL_PATH, map_location=torch.device(self.device)))
            self.model_name = model

        elif model == "T5":

            if not os.path.isfile(T5_MODEL_PATH):
                # raise Exception(
                #   "trained model is not found , either retrain or download")
                # TODO: refine
                logger.info("file not found so downloading")
                GenericUtils.s3downloader("", "", "")

            from transformers import (T5ForConditionalGeneration, T5Tokenizer)
            model_config = GenericUtils.load_config(T5_MODEL_JSON)
            self.tokenizer = T5Tokenizer.from_pretrained(
                model_config['tokenizer'])
            self.model = T5ForConditionalGeneration.from_pretrained(
                T5_MODEL_PATH)
        self.model.to(self.device)  # magic !
        self.model.eval()

    """
        This method is called implicitly ! to predict sentiment.

        :param sentence: sentence to predict sentiment on
        :type: string.

    """

    def predict(self, sentence):

        if self.model_name == "bert":
            tokens = self.tokenizer.tokenize(sentence)
            tokens = tokens[:self.tokenizer.max_input_length - 2]
            index = [self.tokenizer.init_token_idx]
            self.tokenizer.tokenizer.convert_tokens_to_ids(
                tokens) + [self.tokenizer.eos_token_idx]
            torch_token_tensor = torch.LongTensor(index).to(self.device)
            torch_token_tensor = torch_token_tensor.unsqueeze(0)
            output = torch.sigmoid(self.model(torch_token_tensor))
            score = output.item()
            if score > 0.5:
                return 'postive', str(score)
            else:
                return 'negative', str(score)

        elif self.model_name == "T5":
            input_ids = self.tokenizer.encode(
                sentence, add_special_tokens=True, return_tensors="pt")
            encoded_out = self.model.generate(
                input_ids=input_ids, max_length=2).squeeze()  # batch size 1
            output = self.tokenizer.decode(encoded_out)
            return output

    def __call__(self, *args, **kwargs):
        # TODO: add validation
        return self.predict(args[0])


"""
    summary task

"""


class SummaryTask(Task):

    def __init__(self, sentence, min_length=40, max_length=150):
        super().__init__(sentence)

        self.min_length = min_length
        self.max_length = max_length

        self.token_max_length = 512
        self.model = AutoModelWithLMHead.from_pretrained("t5-base")
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")

        self.num_beams = 4
        self.length_penalty = 2.0

    def predict(self, sentence):

        inputs = self.tokenizer.encode(
            "summarize: " + sentence, return_tensors="pt", max_length=self.token_max_length)
        outputs = self.model.generate(inputs, max_length=self.max_length, min_length=self.min_length,
                                      length_penalty=self.length_penalty, num_beams=self.num_beams, early_stopping=True)
        return outputs

    def __call__(self, *args, **kwargs):
        # TODO: add validation
        return self.predict(args[0])


"""
    Question and Answer Task

"""


class QnATask(Task):

    def __init__(self, sentence, questions):
        super().__init__(sentence)
        self.questions = questions

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad")

    def predict(self, sentence):
        qna = {}
        for question in self.questions:
            inputs = self.tokenizer.encode_plus(
                question, sentence, add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]

            text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            answer_start_scores, answer_end_scores = self.model(**inputs)

            answer_start = torch.argmax(answer_start_scores)
            # Get the most likely beginning of answer with the argmax of the score
            # Get the most likely end of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1

            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

            qna[question] = answer
        return qna

    def __call__(self, *args, **kwargs):
        # TODO: add validation
        return self.predict(args[0])


"""

"""


class NerTask(Task):

    def __init__(self, sentence, questions):
        super().__init__(sentence)
        self.questions = questions

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad")

    def predict(self, sentence):
        qna = {}
        for question in self.questions:
            inputs = self.tokenizer.encode_plus(
                question, sentence, add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]

            text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            answer_start_scores, answer_end_scores = self.model(**inputs)

            answer_start = torch.argmax(answer_start_scores)
            # Get the most likely beginning of answer with the argmax of the score
            # Get the most likely end of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1

            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

            qna[question] = answer
        return qna

    def __call__(self, *args, **kwargs):
        # TODO: add validation
        return self.predict(args[0])


"""
    Any new task, needs to be registered here , includes
    Task-Name and , class which needs to be invoked 

"""

TASK_LIST = {
    "sentiment": {
        "task_class": SentimentTask,
    },
    "summary": {
        "task_class": SummaryTask,
    },
    "qna": {
        "task_class": QnATask,
    },
    "ner": {
        "task_class": "ner_classifier"
    },
    "emotion": {
        "task_class": "emotion_detector"
    }


}

"""
    This is utility class for creating a task and deligating work to it
    PEP 3107

"""


def tasker(task_name: str) -> Task:

    if task_name not in TASK_LIST:
        logging.error("unsupported task {} !, please select from {}".format(
            task_name, list(TASK_LIST.keys())))
        raise KeyError("unsupported task {} !, please select from {}".format(
            task_name, list(TASK_LIST.keys())))

    task_class = TASK_LIST[task_name]['task_class']

    return task_class(task_name)
