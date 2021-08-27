#!/usr/bin/env python3
"""
Modified VGG19 structure for Simpson character classification with 97.3% accuracy

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.


General design of program:

Try easily implemented model first to see if it can solve the problem. If so, change metaparameters to reduce model size and optimize performance
Then try different image transformations to reduce overfitting (balance train accuracy and test accuracy). Finally, change
batch size and use more steps to train final model.

Model selection:

Firstly, I used VGG16 architecture as VGG net use traditional convolution and linear layers, which can easily implemented.
VGG16 has five convolution blocks and three full connected layers. Within
each convolution blocks and fully connected layers, batchnorm is applied to provide performance boost. For
fully connected layers, dropout is set to prevent overfitting. With this modification, the model achieved 96% training accuracy. 

By observing 96% training accuracy as current best training performance for VGG net, I tried Resnet50 as it is more complicated
than VGG net in terms of structure and should achieve better performance. But in experiment, Resnet50 does not provide a performance gain and I
continued with VGG net.


Metrics and optimizer:

I used categorical crossentropy(softmax loss) as loss function since each image has only one class and categorical cross entropy
calculates the distance between predicted classes and one-hot ground truth.
Adam is used to change stepsize as it provides better direction for model by using previous result to guide the current and its 
performance is also proved in training MNIST dataset[1].


Image transformation:

Since the color of images are not important as well as model size is restricted to 50MB, grayscale transformation
is applied to build 1 channel images. Without applying any transformation, my model achieves 96% training accuracy and
66% validation accuracy, which is overfitting on the training data. 

In my perspective, image transformation in training data is used to augment images to
infer the images in test data as well as enhance images. Random horizontal flip, rotation are applied in order to generalize the model. 
Random affine change is used to shear and zoom images as there are character in different sizes and angles in the dataset. 
Random crop is applied after padding the original image as I observed in dataset that there are some images only contain
parts of given character.

Data normalization is applied to both train and test images to prevent large image pixel value affect gradient.

Besides, I tried image enhancement such as histogram equalization but does not give explicit performance boost.
randomPerspective and random vertical flip is also tested but does not give performance boost


Tuning:

Since the input size of original VGG16 is (3,224,224) and image size for 
this problem is (1,64,64), I tried to shrink the network size as the input is not that large. By shrinking 
to 1/4 number of original channels and halve the size of fully connected layer, the model retained same performance.

Then I would like to check whether implementing more layers would result in better performance. I implemented
VGG19 which achieved 98% training accuracy. By intuition, more complex network should achieve better performance.
But when I add more convblocks and more fully connected layers, performance does not increase. 

By applying image transformation, my model only obtain 77% accuracy on the training data but 91% on validation data, which
infers that I am doing incorrect transformations that could harm the model.
To solve this problem, I decreased rotation angles and shear ranges as well as using smaller batch size.


Other steps:
I did not specify initial learning rate and the initial step size is calculated by Adam. This makes the model converge quickly.
Batch size is set to 32 to increase the accuracy of model
I also tried "weight_decay" factor in Adam to add regularization. It does reduce overfitting but also decreased overall performance.
To further boost performance, I used full dataset to train model to obtain a better result when submitting.


Reference:
[1] Kingma, Diederik & Ba, Jimmy. (2014). Adam: A Method for Stochastic Optimization. International Conference on Learning Representations.  link: https://arxiv.org/abs/1412.6980
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomCrop(64, padding=12, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 10)),
            transforms.RandomAffine(0, shear=10, scale=(0.7,1.3)),
            
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            ])
    elif mode == 'test':
        return transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
            ])

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_classes = 14

        # define frequently used functions
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)) 

        # specify layers
        self.layers = [16,32,64,128,256]
        self.linear_layers = [2048,1024,512]

        # first convblock
        self.convlayer1 = self.convnet_generator(1,self.layers[0])
        self.convlayer2 = self.convnet_generator(self.layers[0],self.layers[0])

        # second convblock
        self.convlayer3 = self.convnet_generator(self.layers[0],self.layers[1])
        self.convlayer4 = self.convnet_generator(self.layers[1],self.layers[1])

        # third convblock
        self.convlayer5 = self.convnet_generator(self.layers[1],self.layers[2])
        self.convlayer6 = self.convnet_generator(self.layers[2],self.layers[2])
        self.convlayer7 = self.convnet_generator(self.layers[2],self.layers[2])

        # forth convblock
        self.convlayer8 = self.convnet_generator(self.layers[2],self.layers[3])
        self.convlayer9 = self.convnet_generator(self.layers[3],self.layers[3])
        self.convlayer10 = self.convnet_generator(self.layers[3],self.layers[3])
        self.convlayer14 = self.convnet_generator(self.layers[3],self.layers[3])

        # fifth convblock
        self.convlayer11 = self.convnet_generator(self.layers[3],self.layers[3])
        self.convlayer12 = self.convnet_generator(self.layers[3],self.layers[3])
        self.convlayer13 = self.convnet_generator(self.layers[3],self.layers[3])
        self.convlayer15 = self.convnet_generator(self.layers[3],self.layers[3])

        # Dense layers
        self.linearlayer1 = self.linearnet_generator(512,self.linear_layers[0])
        self.linearlayer3 = self.linearnet_generator(self.linear_layers[0],self.linear_layers[1])
        self.linearlayer4 = self.linearnet_generator(self.linear_layers[1],self.linear_layers[2])
        
        self.outputlayer = nn.Linear(self.linear_layers[2],self.n_classes)

    # used to generate convolution layer and apply batchnorm and relu
    def convnet_generator(self,n_in,n_out):
        layers = [
            nn.Conv2d(in_channels=n_in,out_channels=n_out,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(n_out), # prevent gradient explode and vanish
            nn.ReLU(),
        ]
        return nn.Sequential(*layers)
    
    # used to generate linear layer combined with batchnorm, relu and dropout
    def linearnet_generator(self,n_in,n_out):
        layers = [
            nn.Linear(n_in,n_out),
            nn.BatchNorm1d(n_out), # prevent gradient explode and vanish also improve performance
            nn.ReLU(),
            nn.Dropout(p=0.5) # prevent overfitting        
            ]
        return nn.Sequential(*layers)

    def forward(self, t):

        x = self.convlayer1(t)
        x = self.convlayer2(x)
        x = self.maxpool(x)

        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = self.maxpool(x)

        x = self.convlayer5(x)
        x = self.convlayer6(x)
        x = self.convlayer7(x)
        x = self.maxpool(x)
#------------------------------------
        x = self.convlayer8(x)
        x = self.convlayer9(x)
        x = self.convlayer10(x)
        x = self.convlayer14(x)
        x = self.maxpool(x)

        x = self.convlayer11(x)
        x = self.convlayer12(x)
        x = self.convlayer13(x)
        x = self.convlayer15(x)
        x = self.maxpool(x)

        x = x.view(x.size(0),-1)
        x = self.linearlayer1(x)
        x = self.linearlayer3(x)
        x = self.linearlayer4(x)

        x = self.outputlayer(x)

        return x


net = Network()
lossFunc = nn.CrossEntropyLoss()

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 1
batch_size = 32
epochs = 400
optimiser = optim.Adam(net.parameters())

