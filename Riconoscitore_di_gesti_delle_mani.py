import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# The 'f' at the begining of a print allows the usage of format chars

#DATA PATHS
main_dataset_path="A_Very_Handy_Dataset"
train_path="Training"
test_path="Test"
image_train_path=os.path.join(main_dataset_path,train_path,"sign_mnist_train.csv")
image_test_path=os.path.join(main_dataset_path,test_path,"sign_mnist_test.csv")

IMAGE_SIZE=28
CLASSES=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"] # Only needed for final print

# DEFINE DATASET CLASS
class myDataset(torch.utils.data.Dataset):  
    def __init__(self,image_path):
        self.images_path=image_path
        self.images=[]
        self.labels=[]
        with open(self.images_path,'r') as dataset_file:
            for line in dataset_file.readlines()[1:]: #discard the first row containing the 'names' of the columns
                data=line.split(",")
                label=int(data[0]) # The first "column" contains the known label
                pixels=list(map(int,data[1:])) # map() turns a data of some type into the given type
                image=np.array(pixels,dtype=np.float32).reshape(IMAGE_SIZE,IMAGE_SIZE) # np arrays can use reshape
                image=image/255 #CNNs work better with numbers between 0 and 1
                self.images.append(image)
                self.labels.append(label)
    def __getitem__(self,index): # given an index, return the corresponding image and the label
        image=self.images[index]
        label=self.labels[index]
        return image,label       
    def __len__(self):
        return len(self.labels)

# MODEL DEFINITION
# Chosen model: AlexNet
class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet,self).__init__()
        # exctracts features from an image
        self.features=nn.Sequential(
            #N.B batch normalization normalizes between [0-1] an entire batch; added cuz seems to be good; param must be equal to the number of filters in previous Conv2d
            #N.B MaxPool2d Considers a 3x3 kernel and only keeps the max value, so the output filter will be of reduced dim, containing only max vaules
            #1° CONV
            nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=1),
            #2° CONV
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            #3° CONV
            nn.Conv2d(128,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        #classifica
        #use computed features to extract the correct class
            # formula per ottenere le dimensioni in output di un layer convoluzionale: [(size−Kernel+2*Padding)/Stride]+1
        self.classifier=nn.Sequential(
            nn.Dropout(), # reduces overfitting, nullyfing some random nodes, so the model wont get used to the same nodes
            nn.Linear(9216,1024), # 
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,26) # 26 possible output classes
        )
    def forward(self,im):
        x=self.features(im)
        print("features shape:",x.shape)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x

# TRAIN FUNCTION
def train(epochs,model,optimizer,loss_func,train_loader="",test_loader="",device="cpu"):
    train_losses=[]
    train_acc=[]
    test_acc=[]
    test_losses=[]
    for epoch in range(epochs):
        correct_preds_train=0
        correct_preds_test=0
        print("\tEpoch:",epoch+1,"/",epochs)
        for i,(image,labels) in enumerate(train_loader):
            model.train()
            image=image.to(device)
            labels=labels.to(device)
            #unflatten image
            image=image.view(image.shape[0],1,IMAGE_SIZE,IMAGE_SIZE)
            #image=image.permute(0,3,1,2) # change order of channels
            outputs=model(image) # forward
            # compute number of correct predictions
            _,pred=torch.max(outputs,1) # get the index of the predicted classes, so the secon value raturned by max, discarding the first
            correct=torch.sum(pred==labels) # compute number of correct predictions
            correct_preds_train+=correct.item() # get number as normal python number
            loss=loss_func(outputs,labels)# compute loss
            # backward
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
        #Save loss for successive plots
        train_losses.append(loss.item())
        accuracy=correct_preds_train/len(train_loader.dataset)
        train_acc.append(accuracy)
        print("\t\tTrain:\n\t\t\t-Loss: {:.4f} -Accuracy: {:.3f}".format(loss.item(),accuracy))
        torch.save(model.state_dict(),'model.pt')
        #Test model
        for i,(image,labels) in enumerate(test_loader):
            #model switch to test mode, now dropout does not work

            model.eval()
            with torch.no_grad():
                image=image.to(device)
                labels=labels.to(device)
                image=image.view(image.shape[0],1,IMAGE_SIZE,IMAGE_SIZE)
                #image=image.permute(0,3,1,2) #change order of channels
                outputs=model(image)

                _,pred=torch.max(outputs,1) # get the index of the predicted classes, so the secon value raturned by max, discarding the first
                correct=torch.sum(pred==labels) # compute number of correct predictions
                correct_preds_test+=correct.item() # get number as normal python number
                #compute loss
                loss=loss_func(outputs,labels)
        
        #Save the loss of this epoch to later plot the loss graph
        test_losses.append(loss.item())
        accuracy=correct_preds_test/len(test_loader.dataset)
        test_acc.append(accuracy)
        print("\t\tTest:\n\t\t\t-Loss: {:.4f} -Accuracy: {:.3f}".format(loss.item(),accuracy))
        #Save the model with each epoch, so we can re-use it in successive trainings

    torch.save(model.state_dict(),f'model_{epochs}_epochs.pt') #Save the weigths so we can re-use them without re-train the model
    #plot losses in single image
    train_losses=np.array(train_losses)
    test_losses=np.array(test_losses)

    #new plot
    plt.clf()
    plt.plot(train_losses,label="train_loss")
    plt.legend() #add legend to the plot
    plt.savefig(f"losses_{epochs}_epochs.png") #save the plot as png

    #new plot
    plt.clf()
    plt.plot(train_acc,label="train_acc")
    plt.legend() #add legend to the plot
    plt.savefig(f"accuracy_{epochs}_epochs.png") #save the plot as png
    print("\tPlot completed")

#----------------------MAIN----------------------

def main():
    print("\t_start_setup_menu_choices_placeholder_")
    # MODEL DECLARATION
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use gpu if available
    if device=="cuda": torch.cuda.empty_cache()
    print("\tActive device:",torch.cuda.get_device_name(0))
    model=MyAlexNet().to(device)
    # Choosing loss e optimizer functions
    loss_func=nn.CrossEntropyLoss() # This loss function seems to be more indicated when dealing with classification accuracy problems
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-3) #Adam as optimizer sounds like a good idea
    print("\tModel created")
    # CREATE DATASET
    train_dataset=myDataset(image_train_path) #create a dataset for training
    print("\tTraining dataset created")
    test_dataset=myDataset(image_test_path) #create a dataset for testing
    print("\tTest dataset created")
    # DATALOADER
    batch_size=16
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
    #TRAINING
    epochs=5
    #train(epochs,model,optimizer,loss_func,train_loader,test_loader,device)
    #load model
    model.load_state_dict(torch.load('model_5_epochs.pt'))
    while True:
        index=int(input("insert the index of the image: "))
        with open(image_test_path,'r') as f:
            data=f.readlines()[1:] #discard the first row containing the 'names' of the columns
            if index>len(data):
                print("Index out of range, retry...")
                continue

            data=data[index].split(",")
            pixels=list(map(int,data[1:]))
            
            #reshape pixels to get a normal image 
            image=np.array(pixels,dtype=np.float32).reshape(IMAGE_SIZE,IMAGE_SIZE)
            #upscale image
            upscaled_img=Image.fromarray(image)
            upscaled_img=upscaled_img.resize(size=(400, 400))

            plt.imsave("predicted_sign.png",upscaled_img,cmap="gray")

            #turn numpy array into tensor so we can use torch.unsqueeze
            image=torch.from_numpy(image).to(device) 
            #we need to add 2 more empty dimensions: the first one for the batch size, and the second one for the channels
            image=torch.unsqueeze(torch.unsqueeze(image,dim=0),dim=0) 
            #print("image shape is:",image.shape)
            image=image/255 #CNNs work better with numbers between 0 and 1
            #save image
            
            model.eval()

            softmax=torch.nn.Softmax(dim=1)# use softmax to get the confidence between [0-1] and the sum of all class probabilities to 1
            outputs=softmax(model(image))
            #print("outputs :",outputs)
            confidence,pred=torch.max(outputs,1) # max returns the max value (the % chance of an image representing a given letter) and the index of that letter's class,
            print(f"predicted class: {pred.item()} = {CLASSES[pred.item()]}  with confidence: {confidence.item()}")

#If this file is executed as a secondary module, don't execute main.
#If it's executed as a main file (i.e. using this_file.py) this is executed,
#When executed from an other file, don't call this function
if __name__=="__main__":
    main()
