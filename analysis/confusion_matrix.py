from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    '''
    This function prints and plots the confusion matrix.

    Parameters:
    y_true: array-like of shape (n_samples,)
    y_pred: array-like of shape (n_samples,)
    classes: list of class names corresponding to the labels
    '''
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(xticks_rotation='vertical')
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    
    