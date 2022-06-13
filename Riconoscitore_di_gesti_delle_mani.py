import torch
import torch.nn as nn
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
#DATA PATHS
main_dataset_path="A_Very_Handy_Dataset"
train_path="training"
eval_path="evaluation"
image_train_path=os.path.join(main_dataset_path,train_path,"rgb")
image_eval_path=os.path.join(main_dataset_path,eval_path,"rgb")
k_train_path=os.path.join(main_dataset_path,"training_K.json")
labels_train_path=os.path.join(main_dataset_path,"training_xyz.json")
#Some utilities function
def projectPoints(xyz,K): #function that convert xyz point to uv point
    xyz=np.array(xyz)
    K=np.array(K)
    uv=np.matmul(K,xyz.T).T
    return [uv[0]/uv[2],uv[1]/uv[2]] #divide each element by third element to correctly project into image
def load_file(file_name): #load file from json
    with open(file_name,"r") as json_file: #"with" closes the opened file as soon as the operation in the with scope are all executed
        data=json.load(json_file)
    return data
#Define dataset class
class myDataset(torch.utils.data.Dataset):
    train_split=25000
    def __init__(self,image_path,K_path,labels_path,num_images,subset="train"):
        self.image_path=image_path
        self.subset=subset
        self.num_images=num_images
        if subset=="train":
            #immagini in train non superiori a 25000
            num_images=max(myDataset.train_split,num_images) # We NEED a more elegant way to do this...
            self.images_names=np.sort(os.listdir(image_path))[:num_images]
            self.Ks=np.array(load_file(K_path)[:num_images],dtype=np.float32) #need to specify the type as a 32 bit float, because...?
            self.labels=np.array(load_file(labels_path)[:num_images],dtype=np.float32) #need to specify the type as a 32 bit float, because...?
        elif subset=="eval":
            #immagini in eval mai superiori a quelle rimaste
            num_images=max(num_images,len(os.listdir(image_path))-myDataset.train_split)
            self.images_names=np.sort(os.listdir(image_path))[myDataset.train_split:myDataset.train_split+num_images]
            self.Ks=np.array(load_file(K_path),dtype=np.float32)[myDataset.train_split:myDataset.train_split+num_images]
            self.labels=np.array(load_file(labels_path)[myDataset.train_split:myDataset.train_split+num_images],dtype=np.float32) #need to specify the type as a 32 bit float, because...?
    def __getitem__(self,index): #given an index, return the corresponding image and the label
        image_path=os.path.join(self.image_path,self.images_names[index])
        image=np.array(Image.open(image_path),dtype=np.float32)
        K=self.Ks[index]
        labels=self.labels[index]
        #labels must converted from xyz to uv
        new_labels=[]
        for i in range(len(labels)):
            new_labels.append(projectPoints(labels[i],K)) #append new element to the array
        new_labels=np.array(new_labels) #make array a np.array because its faster and more efficient
        return image,new_labels
    def __len__(self):
        #return len(self.images_names)
        return self.num_images #PERCHE' FATTA COSì (deep reason required)
# MODELs DEFINITION
#formula per ottenere le dimensioni in putput di un layer convoluzionale: [(size−Kernel+2*Padding)/Stride]+1
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #extract features
        self.conv1=nn.Conv2d(3,5,10)
        self.conv2=nn.Conv2d(5,5,5)
        self.conv3=nn.Conv2d(5,5,5)
        self.conv4=nn.Conv2d(5,5,5)
        self.pool=nn.MaxPool2d(2,2)
        #get hand coordinates
        self.fc1=nn.Linear(405,128)
        self.fc3=nn.Linear(128,42)
    def forward(self,im):
        im=im.permute(0,3,1,2) #change order of channels
        x=self.pool(torch.relu(self.conv1(im)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=self.pool(torch.relu(self.conv4(x)))
        x=torch.flatten(x,1) #flatten output; needed because...?
        x=torch.relu(self.fc1(x))
        x=self.fc3(x)
        return x
#Alternative model: AlexNet
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.features=nn.Sequential(
            #1° conv
            nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            #2° conv
            nn.Conv2d(64,192,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            #3° conv
            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            #4° conv
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            #5° conv
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,42)
        )
    def forward(self,im):
        x=self.features(im)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x
#----------------------MAIN----------------------
print("\t_start_setup_menu_choices_placeholder_")
# MODEL DECLARATION
device=torch.device("cuda" if torch.cuda.is_available() else "cpu") #Use gpu if available
print("\tActive device:",torch.cuda.get_device_name(0))
#model=CNN().to(device)
model=AlexNet().to(device)
# Choosing loss e optimizer functions
criterion=nn.MSELoss() #Mean Squared Error is general enough to work for us
optimizer=torch.optim.Adam(model.parameters(),lr=1e-5) #Adam as optimizer sounds like a good idea
print("\tModel created")
# CREATE DATASET
train_dataset=myDataset(image_train_path,k_train_path,labels_train_path,10000,subset="train") #create a dataset for training
print("\tTraining dataset created")
eval_dataset=myDataset(image_train_path,k_train_path,labels_train_path,500,subset="eval") #create a dataset for testing
print("\tEval dataset created")
# DATALOADER
batch_size=16
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
eval_loader=torch.utils.data.DataLoader(eval_dataset,batch_size=batch_size,shuffle=True)
# TRAINING
train_losses=[]
eval_losses=[]
epochs=22 # PaulCannon
for epoch in range(epochs):
    print("\tEpoch:",epoch+1,"/",epochs)
    for i,(image,labels) in enumerate(train_loader):
        model.train()
        image=image.to(device)
        labels=labels.to(device)
        image=image.permute(0,3,1,2) ##change order of channels
        outputs=model(image) #forward
        #compute loss
        labels=torch.flatten(labels,start_dim=1)
        loss=criterion(outputs,labels)
        #backward
        optimizer.zero_grad()
        loss.backward()
        #update weights
        optimizer.step()
    #Save loss for successive plots
    train_losses.append(loss.item())
    print("\t\tTrain Loss: {:.2f}".format(loss.item()))
    for i,(image,labels) in enumerate(eval_loader):
        #model switch to evaluation mode, now dropout does not work
        model.eval()
        image=image.to(device)
        labels=labels.to(device)
        image=image.permute(0,3,1,2) #change order of channels
        outputs=model(image)
        #compute loss
        labels=torch.flatten(labels,start_dim=1)
        loss=criterion(outputs,labels)
    #Save the loss of this epoch to later plot the loss graph
    eval_losses.append(loss.item())
    print("\t\tEval Loss: {:.2f}".format(loss.item()))
    #Save the model with each epoch, so we can re-use it in successive trainings
    torch.save(model.state_dict(),'model.pt')
torch.save(model.state_dict(),f'model_{epochs}_epochs.pt') #Save the weigths so we can re-use them without re-train the model
#plot losses in single image
train_losses=np.array(train_losses)
eval_losses=np.array(eval_losses)
plt.plot(train_losses,label="train")
plt.plot(eval_losses,label="eval")
plt.legend() #add legend to the plot
plt.savefig(f"losses_{epochs}_epochs.png") #save the plot as png
print("\tPlot completed")
