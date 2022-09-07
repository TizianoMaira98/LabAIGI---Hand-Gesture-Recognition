# LabAIGI - Hand Gesture Recognition
This program aims to be able to recognize a hand gesture represented in a photo, amongst a number of known hand gestures.
## Packages Required
* PyTorch
* Numpy
* MatPlotLib
* PIL
## Dataset
The dataset used can be found here: https://www.kaggle.com/datasets/datamunge/sign-language-mnist?select=sign_mnist_train
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
