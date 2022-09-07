# LabAIGI - Hand Gesture Recognition
This program is a universitary project that aims to recognize a hand gesture represented in a photo, amongst a number of known hand gestures.
## Packages Required
* PyTorch
* Numpy
* MatPlotLib
* PIL
## Dataset
The dataset used can be found here: https://www.kaggle.com/datasets/datamunge/sign-language-mnist
## Release Info
This handy python script allows the user to classify images showing a hand representing one of the 24 (out of 26) static gesture composing the deaf alphabeth.
Two differently built architectures are provided, to show how this task can be carried how, although with significantly varying perfomances, by similar albeit not equal algorithms
The first is a shortened version of AlexNet, with some changes made through research of state of art procedures
The second contains more convolutional layer, but is based on the Sigmoid activation function, which causes it to perform way worse than the other. An alternative version of this model was tested with Leaky ReLu, but is staying unimplemented, for now.
The program allows the user to choose an arcihtecture, and after that, to follow a series of choices to manipulate the execution through terminal input.
Further improvements on the QoL side may be implemented, one day.
## Performances study
Both models where tested after training with a different number of given epochs, so that we can compare the general performances.
In general, the performances of such type of CNNs may slightly vary with consecutive trainings, because of the randomized choices the models make during some steps
|MyAlexNet Accuracy|MyCNN Accuracy|
|------------------|--------------|
|![5 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Accuracy_Graphs/MyAlexNet_accuracy_5_epochs.png)|![5 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Accuracy_Graphs/MyCNN_accuracy_5_epochs.png)|
|![7 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Accuracy_Graphs/MyAlexNet_accuracy_5_epochs.png)|![7 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Accuracy_Graphs/MyCNN_accuracy_7_epochs.png)|
|![10 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Accuracy_Graphs/MyAlexNet_accuracy_10_epochs.png)|![10 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Accuracy_Graphs/MyCNN_accuracy_10_epochs.png)|
|![12 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Accuracy_Graphs/MyAlexNet_accuracy_12_epochs.png)|![12 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Accuracy_Graphs/MyCNN_accuracy_12_epochs.png)|
|![15 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Accuracy_Graphs/MyAlexNet_accuracy_15_epochs.png)|![15 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Accuracy_Graphs/MyCNN_accuracy_15_epochs.png)|
#
|MyAlexNet Losses|MyCNN Losses|
|----------------|------------|
|![5 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Loss_Graphs/MyAlexNet_losses_5_epochs.png)|![5 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Loss_Graphs/MyCNN_losses_5_epochs.png)|
|![7 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Loss_Graphs/MyAlexNet_losses_7_epochs.png)|![7 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Loss_Graphs/MyCNN_losses_7_epochs.png)|
|![10 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Loss_Graphs/MyAlexNet_losses_10_epochs.png)|![10 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Loss_Graphs/MyCNN_losses_10_epochs.png)|
|![12 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Loss_Graphs/MyAlexNet_losses_12_epochs.png)|![12 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Loss_Graphs/MyCNN_losses_12_epochs.png)|
|![15 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Loss_Graphs/MyAlexNet_losses_15_epochs.png)|![15 epochs](https://raw.githubusercontent.com/TizianoMaira98/LabAIGI---Hand-Gesture-Recognition/master/Loss_Graphs/MyCNN_losses_15_epochs.png)|
