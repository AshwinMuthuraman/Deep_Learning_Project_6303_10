

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
