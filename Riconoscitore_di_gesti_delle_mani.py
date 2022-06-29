import torch
import torch.nn as nn
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt

#DATA PATHS
main_dataset_path="Another_Very_Handy_Dataset"
train_path="Training"
test_path="Test"
image_train_path=os.path.join(main_dataset_path,train_path)
image_test_path=os.path.join(main_dataset_path,test_path)
classes=os.listdir(image_train_path)

# DEFINE DATASET CLASS
class myDataset(torch.utils.data.Dataset):  
    def __init__(self,image_path):
        self.images_path=image_path
        self.images_data=[] # list of couples (path,class)
        classes=os.listdir(image_path) # list of all the classes
        class_index=0 # index of the current class
        # look into each subfolder
        for class_ in classes:
            current_class_images=os.listdir(os.path.join(self.images_path,class_)) # list of all the images path in the current class
            for image in current_class_images:
                complete_image_path=os.path.join(self.images_path,class_,image)
                self.images_data.append([complete_image_path,class_index]) # append a new couple (image path,image class) to the list
            class_index+=1 # next class of images
    def __getitem__(self,index): # given an index, return the corresponding image and the label
        image_path=self.images_data[index][0]
        label=self.images_data[index][1]
        im=Image.open(image_path)
        im=im.resize((1280,720)) # resize image, so we can be sure they all have the same dims
        image=np.array(im,dtype=np.float32) # dtype required cuz yes
        image=image/255.0 #normalize image pixels to [0,1]
        return image,label
    def __len__(self):
        return len(self.images_data) 

# MODEL DEFINITION
# Chosen model: AlexNet
    # formula per ottenere le dimensioni in output di un layer convoluzionale: [(size−Kernel+2*Padding)/Stride]+1
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        #estrae features dell'immagine
        self.features=nn.Sequential(
            #N.B batch normalization added cuz seems to be good; param should/must be equal to he number of filters in Conv2d
            #1° CONV
            nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            #2° CONV
            nn.Conv2d(64,128,kernel_size=5,padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            #3° CONV
            nn.Conv2d(128,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #4° CONV
            nn.Conv2d(64,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        #nel nostro caso non proprio classificatore ma comunque trasforma le feature nel dato che interessa EDIT: cioè?
        self.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(26208,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,5) 
        )
    def forward(self,im):
        x=self.features(im)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x

# TRAIN FUNCTION
def train(epochs,model,optimizer,loss_func,train_loader="",test_loader="",device="cpu"):
    train_losses=[]
    train_acc=[]
    test_losses=[]
    for epoch in range(epochs):
        correct_preds=0
        print("\tEpoch:",epoch+1,"/",epochs)
        for i,(image,labels) in enumerate(train_loader):
            model.train()
            image=image.to(device)
            labels=labels.to(device)
            image=image.permute(0,3,1,2) # change order of channels
            outputs=model(image) # forward
            # compute number of correct predictions
            _,pred=torch.max(outputs,1) # get the index of the predicted classes, so the secon value raturned by max, discarding the first
            correct=torch.sum(pred==labels) # compute number of correct predictions
            correct_preds+=correct.item() # get number as normal python number
            loss=loss_func(outputs,labels)# compute loss
            # backward
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
        #Save loss for successive plots
        train_losses.append(loss.item())
        accuracy=correct_preds/len(train_loader.dataset)
        train_acc.append(accuracy)
        print("\t\tLoss: {:.4f}  Accuracy: {:.3f}".format(loss.item(),accuracy))
        torch.save(model.state_dict(),'model.pt')

        '''
        for i,(image,labels) in enumerate(test_loader):
            #model switch to testuation mode, now dropout does not work
            model.test()
            image=image.to(device)
            labels=labels.to(device)
            image=image.permute(0,3,1,2) #change order of channels
            outputs=model(image)
            #compute loss
            labels=torch.flatten(labels,start_dim=1)
            loss=loss_func(outputs,labels)
        
        #Save the loss of this epoch to later plot the loss graph
        #test_losses.append(loss.item())
        #print("\t\ttest Loss: {:.4f}".format(loss.item()))
        #Save the model with each epoch, so we can re-use it in successive trainings
        '''
    #torch.save(model.state_dict(),f'model_{epochs}_epochs.pt') #Save the weigths so we can re-use them without re-train the model
    #plot losses in single image
    train_losses=np.array(train_losses)
    #test_losses=np.array(test_losses)
    plt.clf()
    plt.plot(train_losses,label="train_loss")
    plt.legend() #add legend to the plot
    plt.savefig(f"losses_{epochs}_epochs.png") #save the plot as png
    plt.clf()
    plt.plot(train_acc,label="train_acc")
    plt.legend() #add legend to the plot
    plt.savefig(f"accuracy_{epochs}_epochs.png") #save the plot as png
    print("\tPlot completed")

#----------------------MAIN----------------------
print("\t_start_setup_menu_choices_placeholder_")
# MODEL DECLARATION
device=torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use gpu if available
if device=="cuda": torch.cuda.empty_cache()
print("\tActive device:",torch.cuda.get_device_name(0))
model=AlexNet().to(device)
# Choosing loss e optimizer functions
loss_func=nn.CrossEntropyLoss() # This loss function seems to be more indicated when dealing with classification accuracy problems
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3) #Adam as optimizer sounds like a good idea
print("\tModel created")
# CREATE DATASET
train_dataset=myDataset(image_train_path) #create a dataset for training
print("\tTraining dataset created")
test_dataset=myDataset(image_train_path) #create a dataset for testing
print("\ttest dataset created")
# DATALOADER
batch_size=16
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=""
#test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
#TRAINING
epochs=15
train(epochs,model,optimizer,loss_func,train_loader,test_loader,device)

'''
#per caricare il modello dal file si usa:
model=AlexNet().to(device) #initialize model
model.load_state_dict(torch.load('model_15_epochs.pt')) #load weigths
i="gino"
while i!="stop":
    i=input("inserisci numero immagine: ")
    visualize_prediction(f"{'0'*(8-len(i))}{i}.jpg",model)
'''
