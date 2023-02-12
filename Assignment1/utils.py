import time
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, ShuffleSplit
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.inputs = torch.tensor(X.to_numpy())
        self.target = torch.tensor(np.where(y == "Y", 1, 0))
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        target = self.target[idx]
        return inputs, target

def load_csv(path):
    """
    Load the CSV form of MNIST data without any external library
    :param path: the path of the csv file
    :return:
        data: A list of list where each sub-list with 28x28 elements
              corresponding to the pixels in each image
        labels: A list containing labels of images
    """
    data = []
    labels = []
    with open(path, 'r') as fp:
        images = fp.readlines()
        images = [img.rstrip() for img in images]

        for img in images:
            img_as_list = img.split(',')
            y = int(img_as_list[0])  # first entry as label
            x = img_as_list[1:]
            x = [int(px) / 255 for px in x]
            data.append(x)
            labels.append(y)
    return data, labels


def load_mnist_trainval():
    """
    Load MNIST training data with labels
    :return:
        train_data: A list of list containing the training data
        train_label: A list containing the labels of training data
        val_data: A list of list containing the validation data
        val_label: A list containing the labels of validation data
    """
    # Load training data
    print("Loading training data...")
    data, label = load_csv('./data/mnist_data/mnist_train.csv')
    assert len(data) == len(label)
    print("Training data loaded with {count} images".format(count=len(data)))

    # split training/validation data
    len_data = len(data) * 0.8
    train_data = data[:int(len_data)]
    train_label = label[:int(len_data)]
    val_data = data[int(len_data):]
    val_label = label[int(len_data):]

    return train_data, train_label, val_data, val_label


def load_mnist_test():
    """
        Load MNIST testing data with labels
        :return:
            data: A list of list containing the testing data
            label: A list containing the labels of testing data
        """
    # Load training data
    print("Loading testing data...")
    data, label = load_csv('./data/mnist_data/mnist_test.csv')
    assert len(data) == len(label)
    print("Testing data loaded with {count} images".format(count=len(data)))

    return data, label

def train(epoch, train_loader, model, optimizer, loss_function, debug=True):
    """
    A training function that trains the model for one epoch
    :param epoch: The index of current epoch
    :param batched_train_data: A list containing batches of images
    :param batched_train_label: A list containing batches of labels
    :param model: The model to be trained
    :param optimizer: The optimizer that updates the network weights
    :return:
        epoch_loss: The average loss of current epoch
        epoch_acc: The overall accuracy of current epoch
    """
    epoch_loss = 0.0
    hits = 0
    count_samples = 0.0

    for idx, data in enumerate(train_loader):
        input, target  = data
        #target = target.unsqueeze(1)
        #target = target.float()
        # zero gradients for every batch
        optimizer.zero_grad()
        #make predictions for this batch
        start_time = time.time()
        pred = model(input.float())
        # compute loss and its gradients
        loss = loss_function(pred, target.reshape(-1,1).float())
        loss.backward()
        # adjust learning weights
        optimizer.step()
        # compute  accuracy
        _, pred_label = torch.max(pred.data, 1)
        correct = (pred_label.reshape(-1) == target.data).sum()
        print(pred_label)
        print(correct)
        print(len(target))
        accuracy = correct/len(target)
        loss = loss.detach().numpy()
        epoch_loss += loss
    
        # count of accurate prediction
        hits += accuracy * len(target)
        # count of total sample data have been trained on
        count_samples += len(target)

        forward_time = time.time() - start_time
        if idx % 100 == 0 and debug:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Batch Time {batch_time:.3f} \t'
                   'Batch Loss {loss:.4f}\t'
                   'Train Accuracy ' + "{accuracy:.4f}" '\t').format(
                epoch, idx, len(train_loader), batch_time=forward_time,
                loss=loss, accuracy=accuracy))

    epoch_loss /= len(train_loader),
    epoch_acc = hits / count_samples

    if debug:
        print("* Average Accuracy of Epoch {} is: {:.4f}".format(epoch, epoch_acc))
    return epoch_loss, epoch_acc


def evaluate(test_loader, model, loss_function, debug=True):
    """
    Evaluate the model on test data
    :param batched_test_data: A list containing batches of test images
    :param batched_test_label: A list containing batches of labels
    :param model: A pre-trained model
    :return:
        epoch_loss: The average loss of current epoch
        epoch_acc: The overall accuracy of current epoch
    """
    epoch_loss = 0.0
    hits = 0
    count_samples = 0.0
    for idx, data in enumerate(test_loader):
        input, target = data
        #target = target.unsqueeze(1)
        #target = target.float()
        pred = model(input.float())
        loss = loss_function(pred, target.reshape(-1,1).float())
        _, pred_label = torch.max(pred.data, 1)
        correct = (pred_label.reshape(-1) == target.data).sum()
        accuracy = correct/len(target)
        loss = loss.detach().numpy()
        epoch_loss += loss
        hits += accuracy * len(target)
        count_samples += len(target)
        if debug:
            print(('Evaluate: [{0}/{1}]\t'
                   'Batch Accuracy ' + "{accuracy:.4f}" '\t').format(
                idx, len(test_loader), accuracy=accuracy))

    epoch_loss /= len(test_loader)
    epoch_acc = hits / count_samples

    return epoch_loss, epoch_acc


def plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, problem_name):
    """
    Plot learning curves with matplotlib. Make sure training loss and validation loss are plot in the same figure and
    training accuracy and validation accuracy are plot in the same figure too.
    :param train_loss_history: training loss history of epochs
    :param train_acc_history: training accuracy history of epochs
    :param valid_loss_history: validation loss history of epochs
    :param valid_acc_history: validation accuracy history of epochs
    :return: None, save two figures in the current directory
    """
    
    print(train_acc_history)
    print(valid_acc_history)
    print(train_loss_history)
    print(valid_loss_history)
    epochs = range(1, 6)
    f = plt.figure(1)
    plt.plot(epochs, train_loss_history, 'g', label = 'Training loss')
    plt.plot(epochs, valid_loss_history, 'b', label = 'Validation loss')
    plt.title(f'Training and Validation loss - {problem_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    f.show()
    plt.savefig(f'Neural_Network_Loss_{problem_name}.png')
    
    g = plt.figure(2)
    plt.plot(epochs, train_acc_history, 'g', label = 'Training accuracy')
    plt.plot(epochs, valid_acc_history, 'b', label = 'Validation accuracy')
    plt.title(f'Training and Validation accuracy - {problem_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    g.show()
    plt.savefig(f'Neural_Network_Accuracy_{problem_name}.png')


def plot_learning_curve(train_data, train_label, est, class_name, problem_name):
    train_sizes, train_scores, validation_scores = learning_curve(est, train_data, train_label)
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.ylabel('MSE', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title(f'Learning curves for {problem_name} - {class_name}', fontsize = 18, y = 1.03)
    plt.legend()
    plt.savefig(f'Learning_curve_{problem_name}_{class_name}.png')





