import pytorch_lightning as pl
import torch
import torchmetrics
from simple_parsing import ArgumentParser
from torch import nn
from torch.nn import functional as F

from config.args import Args

parser = ArgumentParser()
parser.add_arguments(Args, dest="options")
args_namespace = parser.parse_args()
args = args_namespace.options

# Model class
class Model(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)

        n_size = self._get_conv_output(input_shape)

        self.classifier = nn.Linear(n_size, args.num_classes)

    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = self.dropout2(x)
        return x

    def forward(self, x):

        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = Model(input_shape=(3, 224, 224))
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=args.num_classes
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def nll_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.model(x)
        loss = self.nll_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log("accuracy/train_accuracy", acc)
        self.log("loss/train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.model(x)
        loss = self.nll_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log("accuracy/val_accuracy", acc)
        self.log("loss/val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        return optimizer
