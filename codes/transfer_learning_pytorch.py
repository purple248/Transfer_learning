import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim

torch.manual_seed(0)
np.random.seed(0)


# show the images of a batch:
def show_image_batch(img_list, title=None):
    num = len(img_list)
    fig = plt.figure()
    for i in range(num):
        ax = fig.add_subplot(1, num, i+1)
        ax.imshow(img_list[i].numpy().transpose([1,2,0]))
        ax.set_title(title[i])

    plt.show()


# prepares the datasets and dataloaders:
def data_handler(root_dir_train, root_dir_test, batch_size):

    #changing the images sizes, convert t tensor and normelized acording the trained model (RESNET18)
    data_transforms = transforms.Compose([transforms.Resize(256),
                                           torchvision.transforms.RandomCrop((256, 256), padding=None, pad_if_needed=True, fill=0, padding_mode='constant'),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                           ])

    # train data and loader:
    train_dataset = datasets.ImageFolder(root = root_dir_train,
                                         transform = data_transforms)
    train_loader = DataLoader(dataset = train_dataset,
                              batch_size = batch_size,
                              shuffle = True)

    # test data and loader:
    test_dataset = datasets.ImageFolder(root = root_dir_test,
                                         transform = data_transforms)
    test_loader = DataLoader(dataset = test_dataset,
                             batch_size = batch_size,
                             shuffle = True)

    return train_dataset, test_dataset, train_loader, test_loader



# save the model in a different directory:
def save_models(epoch):
    torch.save(model.state_dict(), "../taz_trained_models/taz_model256_{}.model".format(epoch))
    print("Checkpoint saved")

# function for test evaluation:
def test():
    model.eval()
    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader):
        # predict classes using images from the test set
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_acc += torch.sum(prediction == labels.data)

    # compute the average acc and loss over all test images
    test_acc = test_acc / len(test_dataset)

    # to see the prediction to an example of one batch of test images:
    testiter = iter(test_loader)
    images, labels = testiter.next()
    outputs = model(images)
    _, prediction = torch.max(outputs.data, 1)
    indexes = torch.max(outputs,axis = 1).indices
    preds = []
    for index in indexes:
        if index == 0:
            preds.append("taz")
        elif index == 1:
            preds.append("no_taz")

    print(f"true {labels}")
    print(f"preds {indexes} ,{preds}")
    #show_image_batch(images, title=[test_dataset.classes[x] for x in indexes]) #uncomment to see the pictures example

    return test_acc



def train(num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            if i==0 or i//10==10:
                print(f"batch {i}")

            optimizer.zero_grad() # clear the gradients
            outputs = model(images)
            loss = loss_fn(outputs, labels) # compute the loss based on the predictions and actual labels
            loss.backward() # backpropagate the loss
            optimizer.step() # adjust parameters according to the computed gradients

            train_loss += loss.data.item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_acc += torch.sum(prediction == labels.data)


        # adjust the learning rate:
        exp_lr_scheduler.step()

        # compute the average acc and loss over all training images:
        train_acc = train_acc / (len(train_dataset))
        train_loss = train_loss / (len(train_dataset))

        # evaluate on the test set:
        test_acc = test()

        # save the model if the test acc is greater than our current best:
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc

        # print result
        print(f"Epoch {epoch}, Train Accuracy: {train_acc} , TrainLoss: {train_loss} , Test Accuracy: {test_acc}")



# the data directories:
root_dir_train = '../taz_model_data/train'
root_dir_test = '../taz_model_data/test'
batch_size = 4
train_dataset, test_dataset, train_loader, test_loader = data_handler(root_dir_train,root_dir_test,batch_size)

# the model:
# using a trained model - resnet18, that clasiffy 1000 classes
# uploading the trained resnet18 model:
model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
# freezing the weights of this trained model:
for param in model.parameters():
    param.requires_grad = False


# changing the last layer to match the new label dimension (binary: with taz, no taz) - the output dim will be set to 2:
num_input_features = model.fc.in_features
model.fc = nn.Linear(num_input_features, 2)

loss_fn = nn.CrossEntropyLoss() # loss function to minimize
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # Decay LR by a factor of 0.1 every 7 epochs

train(num_epochs = 40) #change the epoch number for more training

