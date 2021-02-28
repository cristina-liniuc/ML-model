import os
import argparse
import itertools
import joblib
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from azureml.core import Dataset, Run






import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import metrics




run = Run.get_context()


def main(args):
    # create the outputs folder
    os.makedirs('outputs', exist_ok=True)

    data_old = pd.read_csv('apartmentComplexData.txt', sep=",", header=None)
    print(data_old.head())

    data = preprocessing.normalize(data_old)
    data = pd.DataFrame(data)
    print(data.head())

    X = data.iloc[:, [2,3,4,5,6]]
    # X = data.iloc[:, :-1].values
    y = data.iloc[:, 8]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_predicted = linear_model.predict(X_test)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_predicted})
    print(df.head())
    print('Mean absolute error: ',metrics.mean_absolute_error(y_test, y_predicted))
    print('Mean squared error: ', metrics.mean_squared_error(y_test, y_predicted))
    print('Mean squared root error: ', np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))
    print("Training set {:.2f}".format(linear_model.score(X_train, y_train)))
    print("Test set {:.2f}".format(linear_model.score(X_test, y_test)))
    
    # files saved in the "outputs" folder are automatically uploaded into run history
    model_file_name = "model.pkl"
    joblib.dump(linear_model, os.path.join('outputs', model_file_name))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=str, default='rbf', help='Kernel type to be used in the algorithm')
    parser.add_argument('--penalty', type=float, default=1.0, help='Penalty parameter of the error term')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args=args)







