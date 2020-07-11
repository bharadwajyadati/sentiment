import torch
import torch.nn as nn
from transformers import BertModel, AlbertModel

"""
    create a custom model for finetuning with pre trained bert model

    after bert pretrained  model  + gru + dropout + fc + softmax

    nn.GRU needs hidden_dim  which can be obtained from bert config  , hidden

    NOTE: Dont modify this class as we are saving the model if names wont match it wont work


"""

__version__ = 0.1


class BertSentimentModel(nn.Module):

    def __init__(self, model='bert'):
        super().__init__()

        self.hidden_dim = 256
        self.output_dim = 1
        self.num_layers = 3
        self.bidirectional = True  # since we have gru ?
        self.dropout = 0.25

        # TODO: load from saved model
        if model == "bert":
            #self.bert = BertModel.from_pretrained()
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            # self.bert.save_pretrained("")
        # albert needs to be implemented
        elif model == "albert":
            self.albert = AlbertModel.from_pretrained("albert-base-v2")

        # emde
        embedding_dim = self.bert.config.to_dict(
        )['hidden_size']  # read the data from bert config for hidden size since we add gru after bert

        # custom GRU layer at end , can we use transformer ? yes , need to work postioninal encoding
        self.rnn = nn.GRU(embedding_dim,
                          hidden_size=self.hidden_dim,
                          num_layers=self.num_layers,
                          bidirectional=self.bidirectional,
                          batch_first=True,
                          dropout=0 if self.num_layers < 2 else self.dropout
                          )
        # one FC layer
        self.out = nn.Linear(
            self.hidden_dim * 2 if self.bidirectional else self.hidden_dim, self.output_dim)

        # dropout layer
        # no softmax ?? smart :) .. its embedded in loss function not as a layer :P
        self.dropout = nn.Dropout(self.dropout)

    """
        forward pass for nn.Module

        adopted from https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66

        self.bert(input_text)[0] --> last_hidden_state --> since pooler_output is not recomended
    """

    def forward(self, input_text):

        # get the embeddings from the bert and use it for GRU
        # note , we dont need to train weights , there are freezed

        with torch.no_grad():
            # taking 1 since needed last_hidden_state not pooler_output
            bert_embedding = self.bert(input_text)[0]

        _, hidden = self.rnn(bert_embedding)

        if self.rnn.bidirectional:  # recomended
            hidden = self.dropout(
                torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        last_layer_ouput = self.out(hidden)

        return last_layer_ouput
