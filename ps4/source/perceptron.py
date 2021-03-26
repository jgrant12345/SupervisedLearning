"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2021 Jan 14
Description : Perceptron
"""

# This code was adapted course material by Tommi Jaakola (MIT).

# data libraries
import numpy as np
import pandas as pd

# scikit-learn libraries
from sklearn.svm import LinearSVC

# matplotlib libraries
import matplotlib.pyplot as plt

######################################################################
# functions
######################################################################

def load_simple_dataset() :
    """Simple dataset of four points."""
    
    #  dataset
    #     i    x^{(i)}        y^{(i)}
    #     1    ( 1,    1)^T   -1
    #     2    ( 0.5, -1)^T    1
    #     3    (-1,   -1)^T    1
    #     4    (-1,    1)^T    1
    #   if outlier is set, x^{(3)} = (12, 1)^T
    
    # data set
    X = np.array([[ 1,    1],
                  [ 0.5, -1],
                  [-1,   -1],
                  [-1,    1]])
    y = np.array([-1, 1, 1, 1])
    return X, y


def plot_data(X, y, ax=None) :
    """Plot features and labels."""
    if ax is None :
        ax = plt.gca()
    pos = np.nonzero(y > 0)  # matlab: find(y > 0)
    neg = np.nonzero(y < 0)  # matlab: find(y < 0)
    ax.plot(X[pos,0], X[pos,1], 'b+', markersize=5)
    ax.plot(X[neg,0], X[neg,1], 'r_', markersize=5)
    #plt.show()


def plot_perceptron(X, y, clf, axes_equal=False, **kwargs) :
    """Plot decision boundary and data."""
    assert isinstance(clf, Perceptron)
    
    # plot options
    if "linewidths" not in kwargs :
        kwargs["linewidths"] = 2
    if "colors" not in kwargs :
        kwargs["colors"] = 'k'
    label = None
    if "label" in kwargs :
        label = kwargs["label"]
        del kwargs["label"]
    
    # hack to skip theta of all zeros
    if len(np.nonzero(clf.coef_)) == 0: return
    
    # axes limits and properties
    xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
    if axes_equal :
        xmin = ymin = min(xmin, ymin)
        xmax = ymax = max(xmax, ymax)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
    
    # create a mesh to plot in
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin, ymax, h))
    
    # plot decision boundary
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = z.reshape(xx.shape)
    cs = plt.contour(xx, yy, zz, [0], **kwargs)
    
    # legend
    if label :
        cs.collections[0].set_label(label)
        plt.legend()


######################################################################
# classes
######################################################################

class Perceptron :
    
    def __init__(self) :
        """
        Perceptron classifier that keeps track of mistakes made on each data point.
        
        Attributes
        --------------------
            coef_     -- numpy array of shape (d,), feature weights
            mistakes_ -- numpy array of shape (n,), mistakes per data point
        """
        self.coef_ = None
        self.mistakes_ = None
    
    def fit(self, X, y, coef_init=None,
            verbose=False, plot=False) :
        """
        Fit the perceptron using the input data.
        
        Parameters
        --------------------
            X         -- numpy array of shape (n,d), features
            y         -- numpy array of shape (n,), targets
            coef_init -- numpy array of shape (d,), initial feature weights
            verbose   -- boolean, for debugging purposes
            plot      -- boolean, for debugging purposes
        
        Returns
        --------------------
            self      -- an instance of self
        """
        # get dimensions of data
        n,d = X.shape
        
        # initialize weight vector to all zeros
        if coef_init is None :
            self.coef_ = np.zeros(d)
        else :
            self.coef_ = coef_init
        
        # record number of mistakes we make on each data point
        self.mistakes_ = np.zeros(n)
        
        # debugging
        if verbose :
            print(f'\ttheta^{{(0)}} = {self.coef_}')
        if plot :
            # set up colors
            colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
            cndx = 1
            
            # plot
            plot_data(X, y)
            plot_perceptron(X, y, self, axes_equal=True,
                            colors=colors[0],
                            label=r"$\theta^{(0)}$")
            
            # pause
            plt.gca().annotate('press key to continue\npress mouse to quit',
                               xy=(0.99,0.01), xycoords='figure fraction', ha='right')
            plt.draw()
            keypress = plt.waitforbuttonpress(0) # True if key, False if mouse
            if not keypress :
                plot = False
        
        ### ========== TODO : START ========== ###
        # part a: implement perceptron algorithm
        # cycle until all examples are correctly classified
        # do NOT shuffle examples on each iteration
        # on a mistake, be sure to update self.mistakes_
        # professor's solution: 10 lines
        
        while not(np.array_equal(y, self.predict(X))):
            for i in range(n):
                if y[i] * np.matmul(np.transpose(self.coef_),X[i]) <= 0:
                    self.coef_ = self.coef_ + y[i] * X[i]
                    self.mistakes_[i] += 1
        
                    # indent the following debugging code to execute every time you update
                    # you can include code both before and after this block
                    mistakes = int(sum(self.mistakes_))
                    if verbose :
                        print(f'\ttheta^{{({mistakes:d})}} = {self.coef_}')
                    if plot :
                        plot_perceptron(X, y, self, axes_equal=True,
                                        colors=colors[cndx],
                                        label=rf"$\theta^{{({mistakes:d})}}$")
                        
                        # set next color
                        cndx += 1
                        if cndx == len(colors) :
                            cndx = 0
                        
                        # pause
                        plt.draw()
                        keypress = plt.waitforbuttonpress(0) # True if key, False if mouse
                        if not keypress :
                            plot = False
        
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X) :
        """
        Predict labels using perceptron.
        
        Parameters
        --------------------
            X         -- numpy array of shape (n,d), features
        
        Returns
        --------------------
            y_pred    -- numpy array of shape (n,), predictions
        """
        return np.sign(np.dot(X, self.coef_))


######################################################################
# main
######################################################################

def main() :
    
    #========================================
    # test part a
    
    # simple data set (from class)
    # coef = [ -1.5, -1], mistakes = 7
    X, y = load_simple_dataset()
    clf = Perceptron()
    clf.fit(X, y, coef_init=np.array([1,0]),
            verbose=True, plot=True)
    print(f'simple data\n\tcoef = {clf.coef_}, mistakes = {int(sum(clf.mistakes_)):d}')
    
    #========================================
    # perceptron data set
    
    train_data = pd.read_csv("../data/perceptron_data.csv", header=None)
    label_col = train_data.columns[-1]
    X_train = train_data.drop([label_col], axis=1).to_numpy()
    y_train = train_data[label_col].to_numpy()
    
    ### ========== TODO : START ========== ###
    # part b: compare different initializations
    # professor's solution: 4 lines


    clf = Perceptron()
    clf.fit(X_train, y_train, coef_init=np.array([0,0]),
            verbose=False, plot=False)
    print(f'simple data\n\tcoef = {clf.coef_}, mistakes = {int(sum(clf.mistakes_)):d}')

    clf = Perceptron()
    clf.fit(X_train, y_train, coef_init=np.array([1,0]),
            verbose=False, plot=False)
    print(f'simple data\n\tcoef = {clf.coef_}, mistakes = {int(sum(clf.mistakes_)):d}')
    
    
    
    ### ========== TODO : END ========== ###
    
    print('perceptron bound')
    
    # you do not have to understand this code -- we will cover it when we discuss SVMs
    # compute gamma using hard-margin SVM (SVM with large C)
    clf = LinearSVC(C=1e10, fit_intercept=False)
    clf.fit(X_train, y_train)
    gamma = 1./np.linalg.norm(clf.coef_, 2)
    
    ### ========== TODO : START ========== ###
    # part c: compare perceptron bound to number of mistakes
    # professor's solution: 4 lines
    
    # compute R^2
    
    # compute perceptron bound (R / gamma)^2
    
    ### ========== TODO : EEND ========== ###

if __name__ == "__main__" :
    main()