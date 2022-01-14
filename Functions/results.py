from scHPL import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def result(true,pred):
    print("Accuracy of test set: ",accuracy_score(true, pred))
    print("F1-score of test set: ",f1_score(true, pred, average="weighted"))
    print("Precision of test set:",precision_score(true, pred, average='weighted'))
    print("Recall of test set:",recall_score(true, pred, average='weighted'))
    print("Classification report of test set:\n",classification_report(true, pred))
    confmatrix = evaluate.confusion_matrix(true, pred)
    confmatrix = confmatrix / np.sum(confmatrix.values, axis = 1, keepdims=True) #Normalize
    plt.figure(figsize=(15,8))
    #sns.light_palette("seagreen", as_cmap=True)
    sns.heatmap(round(confmatrix,2), vmin = 0, vmax = 1, annot=True, cmap='Greens')
    plt.show()