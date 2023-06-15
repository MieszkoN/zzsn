import torch
import torch.nn as nn
import torchvision.models as models

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=200):
        super(CustomResNet18, self).__init__()
        original_model = models.resnet18()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)
        
        # Change filter size to 5
        new_filter_size = 5
        self.features[0] = nn.Conv2d(3, 64, kernel_size=new_filter_size, stride=1, padding=1, bias=False)
        self.to(torch.device("cuda"))


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x