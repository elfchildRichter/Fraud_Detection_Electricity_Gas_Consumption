## Module for metrics in the model part ##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

# training_history['model']
def plot_loss_auc(training_history):
    plt.style.use('bmh')
    plt.rcParams['font.size'] = 16
    
    _, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(training_history.history['loss'], label='loss', color='red')
    ax1.plot(training_history.history['val_loss'], label='val_loss', linestyle='--', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss and AUC')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(training_history.history['auc'], label='auc', color='blue')
    ax2.plot(training_history.history['val_auc'], label='val_auc', linestyle='--', color='blue')
    
    ax2.set_ylabel('AUC')
    ax2.legend()

    plt.show()
    #plt.savefig(f'images/loss_auc_{model}.png')


def print_metrics(name, y_test, y_pred_proba):

    y_pred = y_pred_proba.round()

    print(f'------- Model: {name} -------')
    print('ROC AUC: ', roc_auc_score(y_test, y_pred_proba))
    print('Classification Report: \n', classification_report(y_test, y_pred, digits=3))
    print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
    print('Confusion Matrix (Normalized): \n', confusion_matrix(y_test, y_pred, normalize='true'))
    
    plt.style.use('bmh')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (4, 3))
    plt.grid(False)

    sns.heatmap(confusion_matrix(y_test, y_pred, normalize='true'), annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - model_{name}')
    plt.show()
    #plt.savefig(f'images/confusion_matrix_{name}.png')

    
    plt.rcParams['font.size'] = 16
    plt.figure(figsize = (8, 6))

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'model_{name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.legend()
    plt.show()
    #plt.savefig(f'images/roc_curve_{name}.png')


from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score,fbeta_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def scores_and_matrix(name, y_test, y_pred, y_pred_proba):

    print(f'------- Model: {name} -------')
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Precision: ', precision_score(y_test, y_pred))
    print('Recall: ', recall_score(y_test, y_pred))
    print('F1 Score: ', f1_score(y_test, y_pred))
    print(f'F-Beta: {fbeta_score(y_test, y_pred, beta=2)}')
    print('ROC AUC: ', roc_auc_score(y_test, y_pred_proba))
    print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
    print('Confusion Matrix (Normalized): \n', confusion_matrix(y_test, y_pred, normalize='true'))
    print('Classification Report: \n', classification_report(y_test, y_pred, digits=3))

    plt.style.use('bmh')
    plt.rcParams['font.size'] = 14
    plt.figure(figsize = (4, 3))
    plt.grid(False)

    sns.heatmap(confusion_matrix(y_test, y_pred, normalize='true'), annot=True, cmap='Blues', fmt='.3f')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()
    
    

#input pred_list = [(name1, y_test1, y_pred1, y_pred_proba1), (name2, y_test2, y_pred2, y_pred_proba2), ...]
def all_roc_curves(name_ytest_ypred_ypredproba_list):

    plt.style.use('bmh')
    plt.rcParams['font.size'] = 16
    plt.figure(figsize = (8, 6))
    #plt.axis([0, 1, 0, 1])
    
    plt.plot([0,1], [0,1], 'k--')

    for (name, y_test, y_pred, y_pred_proba) in name_ytest_ypred_ypredproba_list:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=name)
        
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.legend()
    plt.show()


