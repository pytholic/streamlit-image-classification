import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision.models as models
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
    def __init__(self, input_shape, weights=args.weights):
        super().__init__()

        self.feature_extractor = models.resnet18(weights=weights)

        if weights:
            # layers are frozen by using eval()
            self.feature_extractor.eval()
            # freeze params
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        n_size = self._get_conv_output(input_shape)

        self.classifier = nn.Linear(n_size, args.num_classes)

    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.convs(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def convs(self, x):
        x = self.feature_extractor(x)
        return x

    def forward(self, x):

        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = Model(input_shape=args.input_shape)
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=args.num_classes
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def ce_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.model(x)
        loss = self.ce_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log("accuracy/train_accuracy", acc)
        self.log("loss/train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.model(x)
        loss = self.ce_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log("accuracy/val_accuracy", acc)
        self.log("loss/val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "loss/val_loss",
        }
