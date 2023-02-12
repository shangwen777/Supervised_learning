import os
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.nn import functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from utils import train, evaluate, plot_curves, plot_learning_curve, CustomDataset
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
from sklearn.model_selection import train_test_split



parser = argparse.ArgumentParser(description='ml_a1')
parser.add_argument('--config', default='./configs/config.yaml')





def main():

    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    root = './data/load_prediction_dataset'
    if not os.path.exists(root):
        os.mkdir(root)

    data = pd.read_csv("./data/Loan_prediction_dataset/train_loan_prediction.csv")
    print(data.info())
    X = data[["Gender", "Married", "Dependents", "Education", "Self_Employed", "ApplicantIncome"
                ,"CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"]]
    Y = data["Loan_Status"]
    X = pd.get_dummies(X, columns = ['Gender','Married','Dependents','Education','Property_Area','Self_Employed'])
    X.fillna(0, inplace = True)

    train_data, test_data, train_label, test_label = train_test_split(X, Y, test_size = 0.2, random_state = 1)

    train_data_nn = CustomDataset(train_data, train_label)
    test_data_nn = CustomDataset(test_data, test_label)

    # Run Decision Tree 
    start_time = time.time()
    dt_accuracy, L, est_tree = dec_tree(train_data, train_label, test_data, test_label, 'Loan_prediction')
    dt_time = time.time() - start_time
    

    # Run Neural Network
    start_time = time.time()
    nn_accuracy, train_loss_history, train_acc_history, valid_loss_history, valid_acc_history = run(train_data_nn, test_data_nn)
    nn_time = time.time() - start_time
    plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, 'loan_prediction')
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
    knn_accuracy, K, est_knn = knn(train_data, train_label, test_data, test_label, 'loan_prediction')
    knn_time = time.time() - start_time




    print("1 - Decision tree test accuracy:", dt_accuracy, "at L =", L, "running time:", dt_time)
    print("2 - Neural Network test accuracy:", nn_accuracy, "running time:", nn_time)
    print("3 - HistGradientBoosting test accuracy:", gb_accuracy, "running time:", gb_time)
    print("4 - SVM accuracy:", svm_accuracy, "running time:", svm_time)
    print("5 - KNN accuracy:", knn_accuracy, "at K =", K, "running time:", knn_time)
    
    #plot_learning_curve(train_data, train_label, est_tree, 'Decision_tree', 'Loan prediction')
    #plot_learning_curve(train_data, train_label, est_boost, 'Hist_Gradient_boosting', 'Loan prediction')
    #plot_learning_curve(train_data, train_label, est_svm, 'SVM', 'Loan Prediction')
    #plot_learning_curve(train_data, train_label, est_knn, 'KNN', 'Loan prediction')





def run(train_data_nn, test_data_nn):


    train_loader = torch.utils.data.DataLoader(
        dataset=train_data_nn,
        batch_size=args.batch_size,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data_nn,
        batch_size=args.batch_size,
        shuffle=True)

    # Model, Optimizer, Loss
    model = Net(n_input = 20)
    loss_function = nn.BCELoss()
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