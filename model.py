import torch 
import torch.nn.functional as F
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch import nn, optim 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
import torchvision





class NN(pl.LightningModule):
    def __init__(self, input_size, learning_rate, num_classes):
        super().__init__()
        self.lr = learning_rate
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task = 'multiclass', num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
    
    def training_step(self,):
        return 
    
    def validation_step(self,):
        return 
    
    def test_step(self,):
        return
    
    def _common_step(self,):
        return

    def predict_step(self,):
        return 
    
    def configure_optimizers(self,):
        return 



