import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.utils.model_zoo as model_zoo
from torchvision import transforms as T
import torch.nn.functional as F
import torchvision.models as models
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from modules import makeFM
from easydict import EasyDict as edic

class FMNet(nn.Module):
    def __init__(self, num_classes=1, output_stride=16, pretrained=False):
        super(FMNet, self).__init__()
        if output_stride == 16:
            atrous_rates = [6, 12, 18]
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]
        else:
            raise NotImplementedError

        self.backbone = makeFM(output_stride, pretrained)
        self.out = nn.Sequential(
            nn.Conv2d(256, 256, 1,bias=False),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, x):
        input_shape = x.size()[-2:]
        x = self.backbone(x)

        x = self.out(x)
        
        return F.sigmoid(x)
     

    
    
    
#test if the model works
a = torch.rand(2,3,256,256)
model = FMNet()
print(model(a).shape)
