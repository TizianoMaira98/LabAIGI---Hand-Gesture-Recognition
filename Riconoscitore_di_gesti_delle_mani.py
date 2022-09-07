import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from time import process_time
import matplotlib.pyplot as plt
# The 'f' at the begining of a print allows the usage of format chars
# DATA PATHS
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
# MODELs DEFINITION
# Chosen model: AlexNet
class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet,self).__init__()
        self.name="MyAlexNet"
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
            #NB: Dropout() reduces overfitting, nullyfing some random nodes, so the model wont get used to the same nodes
            nn.Dropout(),
            nn.Linear(9216,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,26) # 26 possible output classes
        )
    def forward(self,im):
        x=self.features(im)
        #print("features shape:",x.shape)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x
# Alternative model: MyCNN
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN,self).__init__()
        self.name="MyCNN"
        # exctracts features from an image
        self.features=nn.Sequential(
            #N.B batch normalization normalizes between [0-1] an entire batch; added cuz seems to be good; param must be equal to the number of filters in previous Conv2d
            #N.B MaxPool2d() Considers a 3x3 kernel and only keeps the max value, so the output filter will be of reduced dim, containing only max vaules
            #N.B Sigmoid() seems to be more indicated for binary classification tasks, but whatever
            #1° CONV
            nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=3,stride=1),
            #2° CONV
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=3,stride=1),
            #3° CONV
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=3,stride=1),
            #4° CONV
            nn.Conv2d(64,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=3,stride=1),
            #5° CONV
            nn.Conv2d(32,16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=3,stride=1),
        )
        #classifica
        #use computed features to extract the correct class
            # formula per ottenere le dimensioni in output di un layer convoluzionale: [(size−Kernel+2*Padding)/Stride]+1
        self.classifier=nn.Sequential(
            #NB: Dropout() reduces overfitting, nullyfing some random nodes, so the model wont get used to the same nodes
            nn.Linear(5184,1296),
            nn.Sigmoid(),
            nn.Dropout(),# reduces overfitting, nullyfing some random nodes, so the model wont get used to the same nodes
            nn.Linear(1296,324),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(324,26),# 26 possible output classes
        )
    def forward(self,im):
        x=self.features(im)
        #print("\tDEBUG> features shape:",x.shape)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x
# TRAIN FUNCTION
def train(epochs,model,optimizer,loss_func,train_loader="",test_loader="",device="cpu"):
    start_training=process_time()
    train_losses=[]
    train_accuracy=[]
    test_accuracy=[]
    test_losses=[]
    for epoch in range(epochs):
        correct_preds_train=0
        correct_preds_test=0
        print("\tEpoch:",epoch+1,"/",epochs)
        for i,(image,labels) in enumerate(train_loader):
            model.train()
            image=image.to(device)
            labels=labels.to(device)
            # unflatten image
            image=image.view(image.shape[0],1,IMAGE_SIZE,IMAGE_SIZE)
            #image=image.permute(0,3,1,2) # change order of channels
            torch.autograd.set_detect_anomaly(False)
            outputs=model(image) # forward
            # compute number of correct predictions
            _,pred=torch.max(outputs,1) # get the index of the predicted classes, so the second value raturned by max, discarding the first
            correct=torch.sum(pred==labels) # compute number of correct predictions
            correct_preds_train+=correct.item() # get number as normal python number
            loss=loss_func(outputs,labels) # compute loss
            # backward
            optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(False)
            loss.backward()
            # update weights
            optimizer.step()
        #Save loss for successive plots
        train_losses.append(loss.item())
        accuracy=correct_preds_train/len(train_loader.dataset)
        train_accuracy.append(accuracy)
        print("\t\tTrain:\n\t\t\t-Loss: {:.3f} -Accuracy: {:.3f}".format(loss.item(),accuracy))
        torch.save(model.state_dict(),f'last_saved_{model.name}.pt')#Save the model with each epoch, so we can re-use it in successive trainings
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
        test_accuracy.append(accuracy)
        print("\t\tTest:\n\t\t\t-Loss: {:.3f} -Accuracy: {:.3f}".format(loss.item(),accuracy))
    end_training=process_time()
    print("\tTotal training time:",end_training-start_training,"; Avg Epoch time:",(end_training-start_training)/epochs)
    torch.save(model.state_dict(),f'{model.name}_{epochs}_epochs.pt') #Save the weigths so we can re-use them without re-train the model
    print("\n\tPlotting data graphs...",end='')
    #plot losses in single image
    train_losses=np.array(train_losses)
    test_losses=np.array(test_losses)
    #plot loss
    plt.clf()
    plt.plot(train_losses,label="train_loss")
    plt.plot(test_losses,label="test_loss")
    plt.legend() #add legend to the plot
    plt.savefig(f"{model.name}_losses_{epochs}_epochs.png") #save the plot as png
    #plot accuracy
    plt.clf()
    plt.plot(train_accuracy,label="train_accuracy")
    plt.plot(test_accuracy,label="test_accuracy")
    plt.legend() #add legend to the plot
    plt.savefig(f"{model.name}_accuracy_{epochs}_epochs.png") #save the plot as png
    print("plot completed\n")
#----------------------MAIN----------------------
def main():
    # INITIAL SET-UP And MODEL SELECTION
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use gpu if available
    if device=="cuda": torch.cuda.empty_cache()
    print("\tActive device:",torch.cuda.get_device_name(0))
    #choose cnn architecture. Choice implemented the ugliest way due to laziness
    print("\tChoose a model architecture for the current run.\n\tAvailable CNNs:\n\t- 1) MyAlexNet\n\t- 2) MyCNN")
    while True:
        try:
            model=int(input("\tInsert CNN number: "))
            if model<=0 or model>2:
                print("\tERROR: Choose one of the shown CNN codes, retry...")
                continue
        except ValueError:
            print("\tERROR: Not a number; choose one of the shown CNN codes, retry...")
            continue
        break
    if model==1: model=MyAlexNet().to(device)
    if model==2: model=MyCNN().to(device)    
    # Choosing loss e optimizer functions
    loss_func=nn.CrossEntropyLoss() # This loss function seems to be more indicated when dealing with classification accuracy problems
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-3) # Adam as optimizer sounds like a good idea
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
    while True:
        choice=str(input("\tDo you wish to train the model? [Y-y/N-n or type quit to exit]: "))
        #TRAINING
        if choice=="quit" or choice=="Quit" or choice=="q" or choice=="Q":
            if os.path.exists("predicted_sign.png"): os.remove("predicted_sign.png")
            print("\t\x1B[3mSo Long, and Thanks for All the Fish o7\x1B[0m")
            break
        elif choice=="Y" or choice=="y":
            epochs=int(input("\tInsert number of epochs: "))
            if epochs<1 : epochs=1
            train(epochs,model,optimizer,loss_func,train_loader,test_loader,device)
        elif choice!="N" and choice!="n":
            print("\tAnswer either Y/y or N/n")
            continue
        #TESTING & PLOTTING
        models_list=[model for model in os.listdir('.') if os.path.isfile(model) and model.endswith('.pt')]
        while True: 
            print(f"\n\tAvailable Weights:",*models_list,sep="\n\t\t")
            choice=str(input("\tInsert the name of the weights file you wish to load (type quit to exit): "))
            if choice=="quit" or choice=="Quit" or choice=="q" or choice=="Q":break
            elif os.path.exists(os.fspath(choice))==False:
                print("\n\tFilename must be spelled correctly and comprehend the .pt extension.")
                continue
            else:
                print("\tLoading Weights...\n")
                try:
                    model.load_state_dict(torch.load(choice))#load weights
                except RuntimeError:
                        print("\tERROR: Cannot load this weights file; please make sure the weights were generated by the loaded model, retry...")
                        continue
                with open(image_test_path,'r') as f:
                    data=f.readlines()[1:]#discard the first row containing the 'names' of the columns
                    avg_confidence=0
                    predictions=0
                    while True:
                        try:
                            index=int(input("\tinsert the index of the image (0 to quit): "))
                        except ValueError:
                            print("\tERROR: Image index must be a positive integer in range [1,7172), retry...")
                            continue
                        if index<=0: break
                        elif index>=len(data):
                            print("\tERROR: Index out of range, retry...")
                            continue
                        stringed_image=data[index].split(",")
                        pixels=list(map(int,stringed_image[1:]))
                        image=np.array(pixels,dtype=np.float32).reshape(IMAGE_SIZE,IMAGE_SIZE)#reshape pixels to get a normal image 
                        #upscale image
                        upscaled_img=Image.fromarray(image)
                        upscaled_img=upscaled_img.resize(size=(400,400))
                        plt.imsave("predicted_sign.png",upscaled_img,cmap="gray")
                        image=torch.from_numpy(image).to(device)#turn numpy array into tensor so we can use torch.unsqueeze
                        image=torch.unsqueeze(torch.unsqueeze(image,dim=0),dim=0)#we need to add 2 more empty dimensions: the first one for the batch size, and the second one for the channels
                        image=image/255 #CNNs work better with numbers between 0 and 1
                        model.eval()
                        softmax=torch.nn.Softmax(dim=1)#use softmax to get the confidence between [0-1] and the sum of all class probabilities to 1
                        outputs=softmax(model(image))
                        confidence,pred=torch.max(outputs,1)#max returns the max value (the % chance of an image representing a given letter) and the index of that letter's class,
                        print(f"\tPredicted class: {pred.item()} = {CLASSES[pred.item()]}  with confidence: {confidence.item()}\n")
                        predictions+=1
                        avg_confidence=((avg_confidence*(predictions-1))+confidence.item())/predictions
                    print("\tAvg confidence:",avg_confidence)

#If this file is executed as a secondary module, don't execute main.
#If it's executed as a main file (i.e. using this_file.py) this is executed,
#When executed from an other file, don't call this function
if __name__=="__main__":
    main()
