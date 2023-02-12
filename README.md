# Supervised_learning
This assignment is to implement five types of machine learning algorithms: decision tree, neural network, boosting, support vector machine and k-nearest neighbors. The algorithms will be tested on two self-chosen classification problems: MNIST image classification and loan prediction. 

1. Download the dataset
   a. MNIST dataset can be downloaded through torchvision.datasets
      For example: mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
   b. Loan predication dataset is downloaded from Kaggle: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset?resource=download
2. main.py is to generate testing accuracy and running time for the MNIST classification problem.
   Just run "python main.py" in terminal.
   The code of generating learning curves is commented.
3. main_loan.py is to generate testing accuracy and running time for the loan status prediction problem.
   Just run "python main_loan.py -- config configs/config_loan.ymal" in terminal.
   The code of generating learning curves is commented.

