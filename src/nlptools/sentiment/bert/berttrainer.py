import time
import torch
import logging
import torch.optim as optim
from sentimentmodel import BertSentimentModel


"""
    model trainer
"""


class Trainer(object):

    def __init__(self, train_iterator, validation_iterator):
        # check if gpu is present for faster training
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # learning rate for Adam , needs to be in gridsearch
        self.lr = 0.0003

        # we dont get datasets now !! , only iterators
        self.train_iterator = train_iterator
        self.validation_iterator = validation_iterator

        # save the model with this name , dont change
        self.saved_model = "sentimentmodel.pt"

    """
        returns model trainable parameters in model in crude way

    """

    def model_parameters_count(self, model):
        count = 0
        for parameter in model.parameters:
            if parameter.requires_grad:
                count = count + sum(parameter.numel())
        return count

    """
        this is basically pretraining + gru + softmax

    """

    def train(self):

        extra_layers = []
        epoch_loss = 0
        epoch_accuracy = 0

        model = BertSentimentModel()

        before_freezing_count = self.model_parameters_count(model)
        print('total no of parameters', before_freezing_count)
        logging.info("total no of parameters" + before_freezing_count)

        # freezing or fine-turning code :D , thanks to bert
        for layer_name, parameter in model.named_parameters():
            if layer_name.startswith("bert"):
                parameter.requires_grad = False
            else:
                extra_layers.append(layer_name)

        after_freezing_count = self.model_parameters_count(model)
        print('total no of parameters to be trained', after_freezing_count)
        logging.info("total no of parameters to be trained" +
                     after_freezing_count)

        logging.info("extra new layers added are : " + extra_layers)

        self.optimizer = optim.Adam(
            model.parameters(), lr=self.lr)  # change we change ?

        # loss fuction is binaryclass entrophy with softmax
        self.loss = nn.BCEWithLogitsLoss()  # softmax with binary loss
        # check we have a gpu ?
        model = model.to(self.device)
        self.loss = self.loss.to(self.device)

        # https://discuss.pytorch.org/t/inference-when-output-sigmoid-is-within-bcewithlogitsloss/36522/4

        logging.info(" training started ")
        model.train()

        for batch in self.train_iterator:  # typical training loop batch is data.Field

            self.optimizer.zero_grad()

            preds = model(batch.text).squeeze(1)  # ???
            loss = self.loss(preds, batch.label)  # data .label
            pred_ciel = torch.round(torch.sigmoid(preds))
            true_preds = (pred_ciel == batch.label).float()
            accuracy = true_preds.sum()/len(true_preds)

            # backward propagation

            loss.backward()
            self.optimizer.step()

            # add losses

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

            return epoch_loss / len(self.train_iterator), epoch_accuracy/len(self.train_iterator), model

    def evaluate(self, model):

        epoch_loss = 0
        epoch_accuracy = 0

        model.eval()

        with torch.no_grad():  # no need for evaluation
            for batch in self.validation_iterator:
                preds = model(batch.text).squeeze(1)
                loss = self.loss(preds, batch.label)
                pred_ciel = torch.round(torch.sigmoid(preds))
                true_preds = (pred_ciel == batch.label).float()
                accuracy = true_preds.sum()/len(true_preds)
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()

        return epoch_loss/len(self.validation_iterator), epoch_accuracy / len(self.validation_iterator)

    def training_loop(self, n_epochs):
        best_validation_loss = float('inf')

        for epoch in range(n_epochs):
            start_time = time.time()

            logging.info("training started")
            print("training started")
            train_loss, train_accuracy, model = self.train()

            logging.info("evaluation started")
            print("evaluation started")
            validation_loss, validation_accuray = self.evaluate(model)

            end_time = time.time()

            minutes = int(end_time - start_time / 60)
            seconds = int(end_time - start_time - (minutes * 60))

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                # save best model
                torch.save(model.state_dict(), self.saved_model)

            logging.info("total time for training epoch is " + (epoch + 1))
            logging.info("train loss " + train_loss +
                         " train accuracy " + train_accuracy)
            logging.info("validation loss " + validation_loss +
                         " validation accuracy " + validation_accuray
                         )
            print(
                f'Epoch : {epoch+1:02} | Epoch Time: {minutes}m {seconds}s')
            print(
                'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(
                ' Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    def validation_loop(self):
        model = BertSentimentModel()
        model.load_state_dict(torch.load(self.saved_model))
        test_loss, test_accuracy = self.evaluate(model)

        print("test loss is : " + test_loss +
              " Test Accuracy :" + test_accuracy)
