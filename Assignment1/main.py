import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.nn import functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils import load_mnist_trainval, load_mnist_test, train, evaluate, plot_curves, plot_learning_curve
from decision_tree import dec_tree
from neural_network import Net, LeNet, MLPNet
from boosting import boost, hist_boost
from support_vector_machine import svm
from k_nearest_neighbor import knn
import torch.nn as nn
import argparse
import yaml
import copy
from yaml import Loader
from sklearn import preprocessing
import time
from sklearn.tree import DecisionTreeClassifier



parser = argparse.ArgumentParser(description='ml_a1')
parser.add_argument('--config', default='./configs/config.yaml')

global args
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

root = './data/mnist_data'
if not os.path.exists(root):
    os.mkdir(root)


def main():
    # Load mnist data
    train_data, train_label, val_data, val_label = load_mnist_trainval()
    test_data, test_label = load_mnist_test()
    # Run Decision Tree 
    start_time = time.time()
    dt_accuracy, L, est_tree = dec_tree(train_data, train_label, test_data, test_label, 'mnist')
    dt_time = time.time() - start_time


    # Run Neural Network
    # start_time = time.time()
    nn_accuracy, train_loss_history, train_acc_history, valid_loss_history, valid_acc_history = run()
    nn_time = time.time() - start_time
    plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history)
    # Run Gradient Boosting
    start_time = time.time()
    gb_accuracy, est_boost = hist_boost(train_data, train_label, test_data, test_label)
    gb_time = time.time() - start_time
    # Run Support Vector Machine
    start_time = time.time()
    svm_accuracy, est_svm = svm(train_data, train_label, test_data, test_label)
    svm_time = time.time() - start_time
    # Run K Nearest Neighbor
    train_data = preprocessing.StandardScaler().fit(train_data).transform(train_data)
    test_data = preprocessing.StandardScaler().fit(test_data).transform(test_data)
    start_time = time.time()
    knn_accuracy, K, est_knn = knn(train_data, train_label, test_data, test_label, 'mnist')
    knn_time = time.time() - start_time




    print("1 - Decision tree test accuracy:", dt_accuracy, "at L =", L, "running time:", dt_time)
    print("2 - Neural Network test accuracy:", nn_accuracy, "running time:", nn_time)
    print("3 - HistGradientBoosting test accuracy:", gb_accuracy, "running time:", gb_time)
    print("4 - SVM accuracy:", svm_accuracy, "running time:", svm_time)
    print("5 - KNN accuracy:", knn_accuracy, "at K =", L, "running time:", knn_time)

    
    #plot_learning_curve(train_data, train_label, DecisionTreeClassifier(criterion = "entropy", max_depth = 11))
    #plot_learning_curve(train_data, train_label, est_boost, 'Hist_Gradient_boosting', 'mnist')
    #plot_learning_curve(train_data, train_label, est_svm, 'SVM')
    #plot_learning_curve(train_data, train_label, est_knn, 'KNN', 'mnist')
   





def run():

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(1.0,))])
    # download mnist dataset
    train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True)

    # Model, Optimizer, Loss
    model = MLPNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []
    best_acc = 0.0
    best_model = None

    for epoch in range(args.epochs):

        # Make sure gradient tracking is on, and do pass over the data
        model.train(True)
        epoch_loss, epoch_acc = train(epoch, train_loader, model, optimizer, loss_function, args.debug)


        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        # don't need gradients to do reporting
        model.train(False)

        # evaluate on test data
        valid_loss, valid_acc = evaluate(test_loader, model, loss_function, args.debug)


        if args.debug:
            print("* Validation Accuracy: {accuracy:.4f}".format(accuracy=valid_acc))

        valid_loss_history.append(valid_loss)
        valid_acc_history.append(valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = copy.deepcopy(model)

    _, test_acc = evaluate(test_loader, best_model, loss_function)  # test the best model
 
    if args.debug:
        print("Final Accuracy on Test Data: {accuracy:.4f}".format(accuracy=test_acc))

    return test_acc, train_loss_history, train_acc_history, valid_loss_history, valid_acc_history



if __name__ == '__main__':
    main()