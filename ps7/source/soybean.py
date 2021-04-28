"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2021 Jan 14
Description : Multiclass Classification on Soybean Dataset
              This code was adapted from course material by Tommi Jaakola (MIT)
"""

# python libraries
import os

# data science libraries
import numpy as np
import pandas as pd
import math

# scikit-learn libraries
from sklearn.svm import SVC
from sklearn import metrics

######################################################################
# output code functions
######################################################################

def generate_output_codes(num_classes, code_type) :
    """
    Generate output codes for multiclass classification.
    
    For one-versus-all
        num_classifiers = num_classes
        Each binary task sets one class to +1 and the rest to -1.
        R is ordered so that the positive class is along the diagonal.
    
    For one-versus-one
        num_classifiers = num_classes choose 2
        Each binary task sets one class to +1, another class to -1, and the rest to 0.
        R is ordered so that
          the first class is positive and each following class is successively negative
          the second class is positive and each following class is successively negatie
          etc
    
    Parameters
    --------------------
        num_classes     -- int, number of classes
        code_type       -- string, type of output code
                           allowable: 'ova', 'ovo'
    
    Returns
    --------------------
        R               -- numpy array of shape (num_classes, num_classifiers),
                           output code
    """
    
    ### ========== TODO : START ========== ###
    # part a: generate output codes
    # professor's solution: 13 lines
    # hint: initialize with np.ones(...) and np.zeros(...)
    R = None
    if code_type == 'ova':
        num_classifiers = num_classes
        R = np.full((num_classes, num_classifiers), -1)
        for index in range(num_classes):
            R[index][index] = 1
    else:
        num_classifiers = math.comb(num_classes, 2)
        R = np.ones((num_classes, num_classifiers))
    

    ### ========== TODO : END ========== ###
    
    return R


def load_output_code(filename) :
    """
    Load output code from file.
    
    Parameters
    --------------------
        filename -- string, filename
    """
    
    # load data
    with open(filename, 'r') as fid :
        data = np.loadtxt(fid, delimiter=",")
    
    return data


def test_output_codes():
    R_act = generate_output_codes(3, 'ova')
    R_exp = np.array([[  1, -1, -1],
                      [ -1,  1, -1],
                      [ -1, -1,  1]])    
    assert (R_exp == R_act).all(), "'ova' incorrect"
    
    R_act = generate_output_codes(3, 'ovo')
    R_exp = np.array([[  1,  1,  0],
                      [ -1,  0,  1],
                      [  0, -1, -1]])
    assert (R_exp == R_act).all(), "'ovo' incorrect"


######################################################################
# classes
######################################################################

class MulticlassSVM :
    
    def __init__(self, R, C=1.0, kernel='linear', **kwargs) :
        """
        Multiclass SVM.
        
        Attributes
        --------------------
            R       -- numpy array of shape (num_classes, num_classifiers)
                       output code
            svms    -- list of length num_classifiers
                       binary classifiers, one for each column of R
            classes -- numpy array of shape (num_classes,) classes
        
        Parameters
        --------------------
            R       -- numpy array of shape (num_classes, num_classifiers)
                       output code
            C       -- numpy array of shape (num_classifiers,1) or float
                       penalty parameter C of the error term
            kernel  -- string, kernel type
                       see SVC documentation
            kwargs  -- additional named arguments to SVC
        """
        
        num_classes, num_classifiers = R.shape
        
        # store output code
        self.R = R
        
        # use first value of C if dimension mismatch
        try :
            if len(C) != num_classifiers :
                raise Warning("dimension mismatch between R and C " +
                                "==> using first value in C")
                C = np.ones((num_classifiers,)) * C[0]
        except :
            C = np.ones((num_classifiers,)) * C
        
        # set up and store classifier corresponding to jth column of R
        self.svms = [None for _ in range(num_classifiers)]
        for j in range(num_classifiers) :
            svm = SVC(kernel=kernel, C=C[j], **kwargs)
            self.svms[j] = svm
    
    
    def fit(self, X, y) :
        """
        Learn the multiclass classifier (based on SVMs).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), features
            y    -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        classes = np.unique(y)
        num_classes, num_classifiers = self.R.shape
        if len(classes) != num_classes :
            raise Exception('num_classes mismatched between R and data')
        self.classes = classes    # keep track for prediction
        
        ### ========== TODO : START ========== ###
        # part b: train binary classifiers
        # professor's solution: 13 lines
        
        # HERE IS ONE WAY (THERE MAY BE OTHER APPROACHES)
        #
        # keep two lists, pos_ndx and neg_ndx, that store indices
        #   of examples to classify as pos / neg for current binary task
        #
        # for each class C
        # a) find indices for which examples have class equal to C
        #    [use np.nonzero(CONDITION)[0]]
        # b) update pos_ndx and neg_ndx based on output code R[i,j]
        #    where i = class index, j = classifier index
        #
        # set X_train using X with pos_ndx and neg_ndx
        # set y_train using y with pos_ndx and neg_ndx
        #     y_train should contain only {+1,-1}
        #
        # train the binary classifier
        
        pass
        ### ========== TODO : END ========== ###
    
    
    def predict(self, X) :
        """
        Predict the optimal class.
        
        Parameters
        --------------------
            X         -- numpy array of shape (n,d), features
        
        Returns
        --------------------
            y         -- numpy array of shape (n,), predictions
        """
        
        n,d = X.shape
        num_classes, num_classifiers = self.R.shape
        
        # setup predictions
        y = np.zeros(n)
        
        # discrim_func is a matrix that stores the discriminant function values
        #   row index represents the index of the data point
        #   column index represents the index of binary classifiers
        discrim_func = np.zeros((n,num_classifiers))
        for j in range(num_classifiers) :
            discrim_func[:,j] = self.svms[j].decision_function(X)
        
        # scan through the examples
        for i in range(n) :
            # compute votes for each class
            votes = np.dot(self.R, np.sign(discrim_func[i,:]))
            
            # predict the label as the one with the maximum votes
            ndx = np.argmax(votes)
            y[i] = self.classes[ndx]
        
        return y


######################################################################
# main
######################################################################

def main() :
    # load data
    label_col = 35
    converters = {label_col: ord} # label (column 35) is a character
    
    train_data = pd.read_csv("../data/soybean_train.csv", header=None, converters=converters)
    X_train = train_data.drop([label_col], axis=1).to_numpy()
    y_train = train_data[label_col].to_numpy()
    num_classes = len(set(y_train))
    
    test_data = pd.read_csv("../data/soybean_test.csv", header=None, converters=converters)
    X_test = test_data.drop([label_col], axis=1).to_numpy()
    y_test = test_data[label_col].to_numpy()
    
    # part a : generate output codes
    test_output_codes()
    
    ### ========== TODO : START ========== ###
    # parts b-c : train component classifiers, make predictions,
    #             compare output codes
    # professor's solution: 13 lines
    #
    # use generate_output_codes(...) to generate OVA and OVO codes
    # use load_output_code(...) to load random codes
    #
    # for each output code
    #   train a multiclass SVM on training data and evaluate on test data
    #   setup the binary classifiers using the specified parameters from the handout
    #
    # if you implemented MulticlassSVM.fit(...) correctly,
    #   using OVA
    #   your first trained binary classifier should have
    #   the following indices for support vectors
    #     array([ 12,  22,  29,  37,  41,  44,  49,  55,  76, 134, 
    #            157, 161, 167, 168,   0,   3,   7])
    #   you should find 54 errors on the test data
    
    ### ========== TODO : END ========== ###

if __name__ == "__main__" :
    main()