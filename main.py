import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import glob
import os
import random
from random import shuffle
import cv2 
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import datetime
import time

# PREPARE DATA START #
#To save models in a folder
x_time = datetime.datetime.now()
x_name = x_time.strftime("%d-%m-%Y %H.%M.%S")
x_name = "sonOdev"
modelSavePath = 'Output Data'

#os.makedirs(os.path.join(modelSavePath, x_name))
modelSaveNameA = os.path.join(modelSavePath, x_name, "savedModelA.pth")
f = os.path.join(modelSavePath, x_name, "trainValidAccuA.png")
fig2NameA = os.path.join(modelSavePath, x_name, "trainValidLossA.png")
modelSaveNameG = os.path.join(modelSavePath, x_name, "savedModelG.pth")
fig1NameG = os.path.join(modelSavePath, x_name, "trainValidAccuG.png")
fig2NameG = os.path.join(modelSavePath, x_name, "trainValidLossG.png")
#sonucFileName = os.path.join(modelSavePath, x_name, "sonuclar.txt")


train_folder = glob.glob("./dataset/train/*.jpeg")
test_folder = glob.glob("./dataset/test/*.jpeg")

#dataset size
train_size = 1000
test_size = 200
label_size = 13

#Models input size
height = 224
width = 224

train_dataset = random.sample(train_folder, train_size)
test_dataset = random.sample(test_folder, test_size)

#labels
emptyGrid = 0
blackPawn = 1
whitePawn = 2
blackBishop = 3
whiteBishop = 4
blackRock = 5
whiteRock = 6
blackKnight = 7
whiteKnight = 8
blackQueen = 9
whiteQueen = 10
blackKing = 11
whiteKing = 12

fenToLabel_Dict = {'p' : blackPawn, 'P' : whitePawn,
                  'b' : blackBishop, 'B' : whiteBishop,
                  'r' : blackRock, 'R' : whiteRock,
                  'n' : blackKnight, 'N' : whiteKnight,
                  'q' : blackQueen, 'Q' : whiteQueen,
                  'k' : blackKing, 'K' : whiteKing,
                  '0' : emptyGrid}


# Convert fen notation to corresponding labels
def fenToLabel(fileName):
    
    label = []
    arr=fileName.split('-')

    for grid in range(8):
        row=[]
        for value in arr[grid]:
            if value>='A' and value<='z':
                row.append(fenToLabel_Dict[value])
            elif value >='1' and value<='8':
                for i in range(int(value)):
                    row.append(int(0))
        label = label + row
        
    return label
    
#It is called to convert fens to labels
def prepareLabels(dataset):
    
    Y = []

    for i in range(0, len(dataset)):
        fileName = os.path.splitext(os.path.basename(dataset[i]))[0]
        label = fenToLabel(fileName)
        Y.append(label)
            
    return Y

#Board images are divided into 64 grids
def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

#Images are converted to suitable format for the models
def imagePreprocessing(dataset, h, w):
    
    X = []
    Y = []

    label = prepareLabels(dataset)
    
    for i in range(0, len(dataset)):
        image = cv2.imread(dataset[i])
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_grids = blockshaped(image_gray, 50, 50)
        for j in range(0, len(image_grids)):
          image_resize = cv2.resize(image_grids[j], (h, w))
          image_BGR = cv2.cvtColor(image_resize, cv2.COLOR_GRAY2BGR)
          X.append(image_BGR)
          Y.append(label[i][j])
            
    return X, Y

train_X, train_Y = imagePreprocessing(train_dataset, height, width)
test_X, test_Y = imagePreprocessing(test_dataset, height, width)

#Convert data to tensor
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(
  mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.225]
)])

#create custom dataset for training
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, imageX, labelX):
        self.images = imageX
        self.labels = labelX
        self.transform = transform
    
    def __getitem__(self, index):
        img = self.transform(self.images[index])
        label = self.labels[index]
        return img, label
        
    def __len__(self):
        return len(self.images)
    

custom_dataset = CustomDataset(train_X,train_Y)
test_custom_dataset = CustomDataset(test_X,test_Y)


#Split ratio is 0.2 for validation
train_split = (len(train_X)*8)/10
validation_split = (len(train_X)*2)/10

shuffle_dataset = False
random_seed= 42
batch_size = 128    #Batch size should be determined here

dataset_size = len(custom_dataset)

#Create train and validation datasets and also dataloaders
train_set, validation_set = torch.utils.data.random_split(custom_dataset,[int(train_split),int(validation_split)])
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_custom_dataset, batch_size=batch_size,shuffle=False)

#One batch can be printed
for images, labels in train_loader:
    grid_imgs = torchvision.utils.make_grid(images)
    npimg = grid_imgs.numpy()
    plt.figure(figsize=(30,30))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    print(labels)
    break

# PREPARE DATA END #

# ALEXNET START #

#For model saving into a folder
'''
x_time = datetime.datetime.now()
sonuc_file = open(sonucFileName, "a")
sonuc_file.write("\n\n--------------------\nALEXNET ")
sonuc_file.write(x_time.strftime("%d-%m-%Y %H.%M.%S"))
sonuc_file.write("\n")
'''

from torchvision import models
model = models.alexnet(pretrained=True)
print(model)

label_size = 13

for param in model.parameters():
    param.requires_grad = False

#Finetuning
model.avgpool = nn.AdaptiveAvgPool2d((6, 6))
model.classifier  = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, label_size),
)
print(model)

for name, child in model.named_children():
    print(name)
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    model.cuda()

#Other parameters
AlexNetModel = model
learning_rate = 0.001   #Learning rate should be determined here
momentum = 0.9
weight_decay = 0.0000001
num_epochs  = 10

#Determine cross entropy and params updates
criterion = nn.CrossEntropyLoss()
model = model.to(device)
params_to_update = model.parameters()
print("Params to learn:")
params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)


#Optimizer should be determined here
optimizer = torch.optim.SGD(params_to_update, lr=learning_rate)

#For model saving into a folder
'''
sonuc_file.write("\nLearning Rate: ")
sonuc_file.write(str(learning_rate))
sonuc_file.write("\n")

sonuc_file.write("Optimizer: ")
sonuc_file.write("SGD") # OPTIMIZER GIRMEYI UNUTMA !!!
sonuc_file.write("\n")

sonuc_file.write("Loss: ")
sonuc_file.write("CrossEntropyLoss") # LOSS GIRMEYI UNUTMA !!!
sonuc_file.write("\n")

sonuc_file.write("Batch Size: ")
sonuc_file.write(str(batch_size))
sonuc_file.write("\n\n")
'''

total_step = len(train_loader)
total_images = len(train_dataset*64)
print(f"Number of images: {total_images}, Number of batches: {total_step}")

# helper functions taken from lecture codes, not used
import torch.nn.functional as F

labelToFen_Dict = {0:'0',
                 1:'p',2:'P',
                 3:'b',4:'B',
                 5:'r',6:'R',
                 7:'n',8:'N',
                 9:'q',10:'Q',
                 11:'k',12:'K'}
'''
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def images_to_probs(net, images):
    '''
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(6, 12))

    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            labelToFen_Dict[preds[idx]],
            probs[idx] * 100.0,
            labelToFen_Dict[labels.cpu().numpy()[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig
'''

#Traning Stage
train_losses=[]
train_accu=[]

val_losses=[]
val_accu=[]

steps=0
running_loss=0
print_every=2
total = 0
correct = 0

if torch.cuda.is_available():
    model.cuda()

min_valid_loss = np.inf
for epoch in range (num_epochs):   
    
    tic = time.perf_counter()
    train_accuracy = 0.0
    running_loss=0.0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        
        images = images.cuda()
        labels = labels.cuda()

        steps+=1
        images = images.to(device)
        labels = labels.to(device)


        # Forward pass through the model
        outputs = model(images)

        # Calculate your loss
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        # Make a step with your optimizer
        optimizer.step()

        running_loss+=loss.item()
        
        ps=torch.exp(outputs)
        top_ps, top_class = ps.topk(1, dim=1)
        equality= top_class == labels.view(*top_class.shape)
        train_accuracy+=torch.mean(equality.type(torch.FloatTensor))
        
    #writer.add_figure('predictions vs. actuals',plot_classes_preds(model, images, labels), global_step=epoch * len(train_loader) + i)


    val_loss = 0.0
    val_accuracy = 0.0

    model.eval()     # Optional when not using Model Specific layer
    for i, (images, labels) in enumerate(validation_loader):
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
          
        # Forward Pass
        outputs = model(images)
        # Find the Loss
        loss = criterion(outputs,labels)
        # Calculate Loss
        val_loss+=loss.item()

        #Calculate our accuracy
        ps=torch.exp(outputs)
        top_ps, top_class = ps.topk(1, dim=1)
        equality= top_class == labels.view(*top_class.shape)
        val_accuracy+=torch.mean(equality.type(torch.FloatTensor))


    train_accu.append(train_accuracy/len(train_loader))
    train_losses.append(running_loss/len(train_loader))

    val_accu.append(val_accuracy/len(validation_loader))
    val_losses.append(val_loss/len(validation_loader))
    
    toc = time.perf_counter()
    
    print(f"Epoch: [{epoch+1}/{num_epochs}] "
          f"| Train Loss: {running_loss/len(train_loader):.3f} "
          f"| Train Accuracy: {train_accuracy/len(train_loader):.3f} "
          f"| Val Loss: {val_loss/len(validation_loader):.3f} "
          f"| Val Accuracy: {val_accuracy/len(validation_loader):.3f} "
          f"| Time: {(toc - tic)/60:0.2f} min")
      
    if min_valid_loss > (val_loss/len(validation_loader)):
      print(f"Validation Loss Decreased: {min_valid_loss:.6f} " 
            f"---> {val_loss/len(validation_loader):.6f} ---> Saving Model")

      min_valid_loss = (val_loss/len(validation_loader))

      # Saving State Dict
      torch.save(model.state_dict(), modelSaveNameA)

'''
sonuc_file.write("Train Accuracy\n")
for row in train_accu:
    sonuc_file.write(str(row))
    sonuc_file.write("\n")
    
sonuc_file.write("\nValidation Accuracy\n")
for row in val_accu:
    sonuc_file.write(str(row))
    sonuc_file.write("\n")
    
sonuc_file.write("\nTrain Losses\n")
for row in train_losses:
    sonuc_file.write(str(row))
    sonuc_file.write("\n")
    
sonuc_file.write("\nValidation Losses\n")
for row in val_losses:
    sonuc_file.write(str(row))
    sonuc_file.write("\n")
'''

#Show loss and accuracy plots
plt.figure(dpi=300)
plt.plot(train_accu,'-o')
plt.plot(val_accu,'-o')
plt.grid(color='gray', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Valid'])
plt.title('Train vs Valid Accuracy')

plt.savefig(fig1NameA)

plt.show()

plt.figure(dpi=300)
plt.plot(train_losses,'-o')
plt.plot(val_losses,'-o')
plt.grid(color='gray', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.legend(['Train','Valid'])
plt.title('Train vs Valid Losses')

plt.savefig(fig2NameA)

#Calculate test accuracy
with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        
        # Get predictions and calculate your accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 12800 test images: {} %'.format(100 * correct / total))

'''    
sonuc_file.write('\nAccuracy of the network on the 12800 test images: {} %'.format(100 * correct / total))
sonuc_file.close()
'''

# ALEXNET END #

# ALEXNET POST PROCESSING START # 
#prediction
from torch.autograd import Variable

#labels
emptyGrid = 0
blackPawn = 1
whitePawn = 2
blackBishop = 3
whiteBishop = 4
blackRock = 5
whiteRock = 6
blackKnight = 7
whiteKnight = 8
blackQueen = 9
whiteQueen = 10
blackKing = 11
whiteKing = 12

labelToFen_Dict = {0:"emptyGrid",
                 1:"blackPawn", 2:"whitePawn",
                 3:"blackBishop",4:"whiteBishop",
                 5:"blackRock",6:"whiteRock",
                 7:"blackKnight",8:"whiteKnight",
                 9:"blackQueen",10:"whiteQueen",
                 11:"blackKing",12:"whiteKing"}

labelToFen_Short = {0:"emptyGrid",
                 1:"p", 2:"P",
                 3:"b",4:"B",
                 5:"r",6:"R",
                 7:"n",8:"N",
                 9:"q",10:"Q",
                 11:"k",12:"K"}


test_transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(
  mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.225]
)])

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp = AlexNetModel
mlp.load_state_dict(torch.load(filename))  #Model should be saved before
mlp.eval()

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = mlp(input)
    index = output.data.cpu().numpy().argmax()
    return index

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
imagePath = askopenfilename() # show an "Open" dialog box and return the path to the selected file

# Retrieve item
image_path = []
#image_path.append("E:/Cansu/cansu_spyder/compVision/test/1b1B1Qr1-7p-6r1-2P5-4Rk2-1K6-4B3-8.jpeg")
image_path.append(imagePath)
image, actual_labels = imagePreprocessing(image_path, height, width)


predictions = []

for img in image:
  index = predict_image(img)
  predictions.append(index)

i = 0

liste = []

fig=plt.figure(figsize=(15,15))
for i in range(len(image)):
  # Show result
  index = predict_image(image[i])
  liste.append(index)
  sub = fig.add_subplot(8, 8, i+1)
  sub.set_title(f'Prediction: {index} \n Actual: {actual_labels[i]}', fontsize=10, ha='center')
  plt.axis('off')
  plt.imshow(image[i])
plt.show()


def labelToFen(labelList):
    fenNotation=''
    value=0
    for grid in range(64):
        if grid!=0 and grid%8==0:
            if value!=0:
                fenNotation+=str(value)
            value=0
            fenNotation+='-'
        if labelList[grid]==0:
            value+=1
        else:
            if value!=0:
                fenNotation+=str(value)
            value=0
            fenNotation+=labelToFen_Short[labelList[grid]]
        if grid==63 and value!=0:
            fenNotation+=str(value)
    return fenNotation

print(labelToFen(liste))

# ALEXNET POST PROCESSING END #

# GOOGLENET START # 

'''
x_time = datetime.datetime.now()
sonuc_file = open(sonucFileName, "a")
sonuc_file.write("\n\n--------------------\nGOOGLENET ")
sonuc_file.write(x_time.strftime("%d-%m-%Y %H.%M.%S"))
sonuc_file.write("\n")
'''

import torch
from googlenet_pytorch import GoogLeNet 
model = GoogLeNet.from_pretrained('googlenet')
print(model)

for param in model.parameters():
    param.requires_grad = False

#finetuning
model.aux1.conv.conv = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
model.aux1.bn = nn.BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
model.aux1.fc1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
model.aux1.fc2 = nn.Linear(in_features=1024, out_features=1000, bias=True)

model.aux2.conv.conv = nn.Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
model.aux2.bn = nn.BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
model.aux2.fc1 = nn.Linear(in_features=2048, out_features=1024, bias=True)
model.aux2.fc2 = nn.Linear(in_features=1024, out_features=1000, bias=True)

model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
model.dropout = nn.Dropout(p=0.2, inplace=False)
model.fc = nn.Linear(in_features=1024, out_features=label_size, bias=True)

for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    model.cuda()

GoogleNetModel = model
model.aux_logits = True
learning_rate = 0.1     #learning rate should be determined here
momentum = 0.9
weight_decay = 0.0000001
num_epochs  = 10

#Determine cross entropy and params updates
criterion = nn.CrossEntropyLoss()
model = model.to(device)
params_to_update = model.parameters()
print("Params to learn:")

#parameters update
params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)

#optimizer should be bdetermined here
optimizer = torch.optim.Adam(params_to_update, lr=learning_rate)

'''
sonuc_file.write("\nLearning Rate: ")
sonuc_file.write(str(learning_rate))
sonuc_file.write("\n")

sonuc_file.write("Optimizer: ")
sonuc_file.write("Adam") # OPTIMIZER GIRMEYI UNUTMA !!!
sonuc_file.write("\n")

sonuc_file.write("Loss: ")
sonuc_file.write("CrossEntropyLoss") # LOSS GIRMEYI UNUTMA !!!
sonuc_file.write("\n")

sonuc_file.write("Batch Size: ")
sonuc_file.write(str(batch_size))
sonuc_file.write("\n\n")
'''

total_step = len(train_loader)
total_images = len(train_dataset*64)
print(f"Number of images: {total_images}, Number of batches: {total_step}")

#training stage
train_losses=[]
train_accu=[]

val_losses=[]
val_accu=[]

steps=0
running_loss=0
print_every=2
total = 0
correct = 0

if torch.cuda.is_available():
    model.cuda()

min_valid_loss = np.inf
for epoch in range (num_epochs):   
    
    tic = time.perf_counter()
    train_accuracy = 0.0
    running_loss=0.0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        
        
        images = images.cuda()
        labels = labels.cuda()

        steps+=1
        images = images.to(device)
        labels = labels.to(device)
        

        # Forward pass through the model
        outputs, aux1, aux2  = model(images)
        
        # Calculate your loss
        loss = criterion(outputs, labels) + 0.3 * (criterion(aux1, labels) + criterion(aux2, labels))
        
        optimizer.zero_grad()
        loss.backward()

        # Make a step with your optimizer
        optimizer.step()

        running_loss += loss.item()
        
        ps = torch.exp(outputs)
        top_ps, top_class = ps.topk(1, dim=1)
        equality= top_class == labels.view(*top_class.shape)
        train_accuracy += torch.mean(equality.type(torch.FloatTensor))
        
    #writer.add_figure('predictions vs. actuals',plot_classes_preds(model, images, labels), global_step=epoch * len(train_loader) + i)

    val_loss = 0.0
    val_accuracy = 0.0

    model.eval()     # Optional when not using Model Specific layer
    
    for i, (images, labels) in enumerate(validation_loader):
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
          
        # Forward Pass
        outputs = model(images)
        # Find the Loss
        loss = criterion(outputs, labels)
        # Calculate Loss
        val_loss+=loss.item()

        #Calculate our accuracy
        ps=torch.exp(outputs)
        top_ps, top_class = ps.topk(1, dim=1)
        equality= top_class == labels.view(*top_class.shape)
        val_accuracy+=torch.mean(equality.type(torch.FloatTensor))


    train_accu.append(train_accuracy/len(train_loader))
    train_losses.append(running_loss/len(train_loader))

    val_accu.append(val_accuracy/len(validation_loader))
    val_losses.append(val_loss/len(validation_loader))

    toc = time.perf_counter()
    print(f"Epoch: [{epoch+1}/{num_epochs}] "
          f"| Train Loss: {running_loss/len(train_loader):.3f} "
          f"| Train Accuracy: {train_accuracy/len(train_loader):.3f} "
          f"| Val Loss: {val_loss/len(validation_loader):.3f} "
          f"| Val Accuracy: {val_accuracy/len(validation_loader):.3f} "
          f"| Time: {(toc - tic)/60:0.2f} min")
      
    if min_valid_loss > (val_loss/len(validation_loader)):
      print(f"Validation Loss Decreased: {min_valid_loss:.6f} " 
            f"---> {val_loss/len(validation_loader):.6f} ---> Saving Model")
      min_valid_loss = (val_loss/len(validation_loader))

      # Saving State Dict
      torch.save(model.state_dict(), modelSaveNameG)

'''
sonuc_file.write("Train Accuracy\n")
for row in train_accu:
    sonuc_file.write(str(row))
    sonuc_file.write("\n")
    
sonuc_file.write("\nValidation Accuracy\n")
for row in val_accu:
    sonuc_file.write(str(row))
    sonuc_file.write("\n")
    
sonuc_file.write("\nTrain Losses\n")
for row in train_losses:
    sonuc_file.write(str(row))
    sonuc_file.write("\n")
    
sonuc_file.write("\nValidation Losses\n")
for row in val_losses:
    sonuc_file.write(str(row))
    sonuc_file.write("\n")
'''

#Plot loss and accuracy
plt.figure(dpi=300)
plt.plot(train_accu,'-o')
plt.plot(val_accu,'-o')
plt.grid(color='gray', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Valid'])
plt.title('Train vs Valid Accuracy')

plt.savefig(fig1NameG)

plt.show()

plt.figure(dpi=300)
plt.plot(train_losses,'-o')
plt.plot(val_losses,'-o')
plt.grid(color='gray', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.legend(['Train','Valid'])
plt.title('Train vs Valid Losses')

plt.savefig(fig2NameG)

plt.show()

#Calculate test accuracy
with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        # If you don't use no_grad context you can use
        # model.eval() function
        # When you use it your model enters to evaluation mode (no grad calculation) 
        # Be careful some layers (BatchNorm) behaves different in training and evaluation mode 
        # You know we calculate local gradients when we do forward pass
        
        outputs = model(images)
        
        # Get predictions and calculate your accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 12800 test images: {} %'.format(100 * correct / total))
 
'''
sonuc_file.write('\nAccuracy of the network on the 12800 test images: {} %'.format(100 * correct / total))
sonuc_file.close()
'''

# GOOGLENET END #

# GOOGLENETMODEL POST PROCESSING START #
#prediction
from torch.autograd import Variable

#labels
emptyGrid = 0
blackPawn = 1
whitePawn = 2
blackBishop = 3
whiteBishop = 4
blackRock = 5
whiteRock = 6
blackKnight = 7
whiteKnight = 8
blackQueen = 9
whiteQueen = 10
blackKing = 11
whiteKing = 12

labelToFen_Dict = {0:"emptyGrid",
                 1:"blackPawn", 2:"whitePawn",
                 3:"blackBishop",4:"whiteBishop",
                 5:"blackRock",6:"whiteRock",
                 7:"blackKnight",8:"whiteKnight",
                 9:"blackQueen",10:"whiteQueen",
                 11:"blackKing",12:"whiteKing"}

labelToFen_Short = {0:"emptyGrid",
                 1:"p", 2:"P",
                 3:"b",4:"B",
                 5:"r",6:"R",
                 7:"n",8:"N",
                 9:"q",10:"Q",
                 11:"k",12:"K"}

test_transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(
  mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.225]
)])

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp = GoogleNetModel
mlp.load_state_dict(torch.load(filename))
mlp.eval()


def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = mlp(input)
    index = output.data.cpu().numpy().argmax()
    return index

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
imagePath = askopenfilename() # show an "Open" dialog box and return the path to the selected file

# Retrieve item
image_path = []
#image_path.append("E:/Cansu/cansu_spyder/compVision/test/1b1B1Qr1-7p-6r1-2P5-4Rk2-1K6-4B3-8.jpeg")
image_path.append(imagePath)
image, actual_labels = imagePreprocessing(image_path, height, width)

predictions = []

for img in image:
  index = predict_image(img)
  predictions.append(index)

i = 0

liste = []

fig=plt.figure(figsize=(15,15))
for i in range(len(image)):
  # Show result
  index = predict_image(image[i])
  liste.append(index)
  sub = fig.add_subplot(8, 8, i+1)
  sub.set_title(f'Prediction: {index} \n Actual: {actual_labels[i]}', fontsize=10, ha='center')
  plt.axis('off')
  plt.imshow(image[i])
plt.show()


def labelToFen(labelList):
    fenNotation=''
    value=0
    for grid in range(64):
        if grid!=0 and grid%8==0:
            if value!=0:
                fenNotation+=str(value)
            value=0
            fenNotation+='-'
        if labelList[grid]==0:
            value+=1
        else:
            if value!=0:
                fenNotation+=str(value)
            value=0
            fenNotation+=labelToFen_Short[labelList[grid]]
        if grid==63 and value!=0:
            fenNotation+=str(value)
    return fenNotation

print(labelToFen(liste))


# FINAL #
'''
sonuc_file.close()



dataFileName = os.path.join(modelSavePath, x_name, "dataset4.txt")
data_file = open(dataFileName, "a")

data_file.write("Train Folder\n")
for row in train_folder:
    data_file.write(str(row))
    data_file.write("\n")

data_file.close()
'''
