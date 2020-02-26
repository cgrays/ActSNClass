"""
================
Confusion matrix
================

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(cm = None, y_true = None, y_pred = None, classes = None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          size=None,
                          plot=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    # Compute confusion matrix
    if cm is None:
        cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if classes is None:        
        if y_true is not None:
            classes = unique_labels(y_true, y_pred)     
        else:            
            if cm.shape == (8,8):
                classes = ['II', 'Ia', 'Ia-91bg', 'Ia-x', 'Ibc', 'KN', 'SLSN-I', 'TDE']
            elif cm.shape == (3,3):
                classes = ['II', 'Ia', 'Ibc']
       
    if plot:
    
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        #print(cm)

        if size is not None:
            fig, ax = plt.subplots(figsize=size)
        else:
            fig,ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        cbar = ax.figure.colorbar(im, ax=ax)
        if normalize:
            cbar.set_clim(0.,1.)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes if classes is not None else [],
               yticklabels=classes if classes is not None else [],
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        #fig.show()
        return ax
    
    else:
        return cm

