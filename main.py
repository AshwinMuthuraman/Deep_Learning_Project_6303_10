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
