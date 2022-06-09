import torch
import torch.nn as nn
import numpy as np
import json
import os
from PIL import Image
#DATA PATHS
main_dataset_path="A_Very_Handy_Dataset"
train_path="training"
eval_path="evaluation"
image_train_path=os.path.join(main_dataset_path,train_path,"rgb")
image_eval_path=os.path.join(main_dataset_path,eval_path,"rgb")
k_train_path=os.path.join(main_dataset_path,"training_K.json")
labels_train_path=os.path.join(main_dataset_path,"training_xyz.json")
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
    def __init__(self,image_path,K_path,labels_path,num_images):
        self.image_path=image_path
        self.images_names=np.sort(os.listdir(image_path))[:num_images]
        self.Ks=np.array(load_file(K_path)[:num_images],dtype=np.float32) #need to specify the type as a 32 bit float, because...?
        self.labels=np.array(load_file(labels_path)[:num_images],dtype=np.float32) #need to specify the type as a 32 bit float, because...?
    def __getitem__(self,index): #given an index, return the corresponding image and the label
        image_path=os.path.join(self.image_path,self.images_names[index])
        image=np.array(Image.open(image_path),dtype=np.float32)
        K=self.Ks[index]
        labels=self.labels[index]
        #labels must converted from xyz to uv
        new_labels=[]
        for i in range(len(labels)):
            new_labels.append(projectPoints(labels[i],K)) #append new element to the array
        new_labels=np.array(new_labels) #make array a np.array because its better because...?
        return image,new_labels
    def __len__(self):
        return len(self.images_names)
dataset=myDataset(image_train_path,k_train_path,labels_train_path,1000) #create a dataset
#Dataloader
batch_size=16
train_loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
#MODEL
#formula to get output_size of cnn layer: [(size−Kernel+2*Padding)/Stride]+1
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(3,5,10)
        self.pool=nn.MaxPool2d(2,2) #serve per rimpicciolire l'im
        self.conv2=nn.Conv2d(5,5,5)
        self.conv3=nn.Conv2d(5,5,5)
        self.conv4=nn.Conv2d(5,5,5)
        self.fc1=nn.Linear(405,128)
        self.fc3=nn.Linear(128,42)
    def forward(self,im):
        #change order of channels
        im=im.permute(0,3,1,2)
        x=self.pool(torch.relu(self.conv1(im)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=self.pool(torch.relu(self.conv4(x)))
        x=torch.flatten(x,1) #flatten output; needed because...?
        #print(len(x))
        x=torch.relu(self.fc1(x))
        x=self.fc3(x)
        return x
net=CNN()
criterion=nn.MSELoss() #define/choose loss function: MSE. Other options? What's better AND WHY?
optimizer=torch.optim.Adam(net.parameters(),lr=0.0001) #define/choose optimizer function: MSE. Other options? What's better AND WHY?
#Train the model
epochs=10
for epoch in range(epochs):
    for i,(image,labels) in enumerate(train_loader):
        #forward
        outputs=net(image)
        #print(outputs)
        #compute loss
        labels=torch.flatten(labels,start_dim=1)
        #print(labels.size())
        loss=criterion(outputs,labels)
        #backward
        optimizer.zero_grad()
        loss.backward()
        #update weights
        optimizer.step()
    print("Epoch:",epoch,"/",epochs,"Loss:",loss.item())
