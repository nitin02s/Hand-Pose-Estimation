## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import torchvision.models as models
import numpy as np




class CNNBase(nn.Module):
    def __init__(self):
        super().__init__()

    def training_step(self, batch):
        images, keypoints=batch
        images=images.float()
        keypoints=keypoints.float()
        pred_keypoints=self(images)
        # print(type(keypoints), type(pred_keypoints))
        loss=F.smooth_l1_loss(keypoints, pred_keypoints)
        #loss=loss.float()
        return loss

    def validation_step(self, batch):
        img, label=batch
        preds=self(img)
        loss=F.smooth_l1_loss(preds, label)
        #loss=loss.float()
        return {"val_loss": loss.detach()}

    def validation_per_epoch(self, val_step_output):
        batch_losses=[i["val_loss"] for i in val_step_output]
        epoch_loss=torch.stack(batch_losses).mean()
        # batch_accs=[i["val_acc"] for i in val_step_output]
        # epoch_accs=torch.stack(batch_accs).mean()
        return {"val_loss": epoch_loss.item()}


class Net(CNNBase):

    def __init__(self):
        super().__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 32, 5)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32,64,3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64,128,3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128,256,3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256,512,1)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512*6*6, 1024)
        self.fc2 = nn.Linear(1024, 42)



        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.25)
        self.drop4 = nn.Dropout(p = 0.25)
        self.drop5 = nn.Dropout(p = 0.3)
        self.drop6 = nn.Dropout(p = 0.4)


    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.drop4(x)
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.drop5(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop6(x)
        x = self.fc2(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x


class Net2(CNNBase):
    def __init__(self):
        super().__init__()
        self.network=models.resnet34(pretrained=True)
        n=self.network.fc.in_features
        self.network.fc=nn.Linear(n, 42)

    def forward(self, xb):
        return nn.functional.relu(self.network(xb))

    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True

    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True
