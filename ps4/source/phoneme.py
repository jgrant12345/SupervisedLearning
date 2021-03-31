"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2021 Jan 14
Description : Perceptron vs Logistic Regression on a Phoneme Dataset
"""

import itertools

# data libraries
import numpy as np
import pandas as pd

# scipy libraries
from scipy import stats

# scikit-learn libraries
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Perceptron, LogisticRegression

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

######################################################################
# functions
######################################################################

def get_classifier(clf_str) :
    """
    Initialize and return a classifier represented by the given string.
    
    Parameters
    --------------------
        clf_str     -- string, classifier name
                       implemented
                           "dummy"
                           "perceptron"
                           "logistic regression"
    
    Returns
    --------------------
        clf         -- scikit-learn classifier
        param_grid  -- dict, hyperparameter grid
                           key = string, name of hyperparameter
                           value = list, parameter settings to search
    """
    
    if clf_str == "dummy" :
        clf = DummyClassifier(strategy='stratified')
        param_grid = {}
    elif clf_str == "perceptron" :
        ### ========== TODO : START ========== ###
        # part b: modify two lines below to set parameters for perceptron
        # classifier parameters
        #   estimate intercept, use L2-regularization, set max iterations of 10k
        # parameter search space
        #   find the parameter for tuning regularization strength
        #   let search values be [1e-5, 1e-4, ..., 1e5] (hint: use np.logspace)
        
        clf = Perceptron(penalty = 'l2' ,fit_intercept = True, max_iter = 10000)
        param_grid = {'alpha': np.logspace(-5,5,num=11)}
        ### ========== END : START ========== ###
    elif clf_str == "logistic regression" :
        ### ========== TODO : START ========== ###
        # part b: modify two lines below to set parameters for logistic regression
        # classifier parameters
        #     estimate intercept, use L2-regularization and lbfgs solver, set max iterations of 10k
        # parameter search space
        #    find the parameter for tuning regularization strength
        #    let search values be [1e-5, 1e-4, ..., 1e5] (hint: use np.logspace)
        
        clf = LogisticRegression(fit_intercept = True, penalty = 'l2', solver = 'lbfgs', max_iter = 10000)
        param_grid = {'C': np.logspace(-5,5,num=11)}
        ### ========== END : START ========== ###
    
    return clf, param_grid


def get_performance(clf, param_grid, X, y, ntrials=100) :
    """
    Estimate performance used nested 5x2 cross-validation.
    
    Parameters
    --------------------
        clf             -- scikit-learn classifier
        param_grid      -- dict, hyperparameter grid
                               key = string, name of hyperparameter
                               value = list, parameter settings to search
        X               -- numpy array of shape (n,d), features values
        y               -- numpy array of shape (n,), target classes
        ntrials         -- integer, number of trials
        
    Returns
    --------------------
        train_scores    -- numpy array of shape (ntrials,), average train scores across cv splits
                           scores computed via clf.score(X,y), which measures accuracy
        test_scores     -- numpy array of shape (ntrials,), average test scores across cv splits
                           scores computed via clf.score(X,y), which measures accuracy
    """
    
    train_scores = np.zeros(ntrials)
    test_scores = np.zeros(ntrials)
    
    ### ========== TODO : START ========== ###
    # part c: compute average performance using 5x2 cross-validation
    # hint: use StratifiedKFold, GridSearchCV, and cross_validate
    # professor's solution: 6 lines

    for i in range(ntrials):

        inner_cv = StratifiedKFold(n_splits= 2, shuffle=True, random_state= i)
        outer_cv = StratifiedKFold(n_splits= 5, shuffle=True, random_state= i)

        # Non_nested parameter search and scoring
        model = GridSearchCV(estimator = clf, param_grid=param_grid, cv=inner_cv)
        
        model.fit(X,y)

        # Nested CV with parameter optimization
        nested_score = cross_validate(model, X = X, y = y, cv=outer_cv,return_train_score = True)
        test_scores[i] = nested_score['test_score'].mean()
        train_scores[i] = nested_score['train_score'].mean()
        # test_scores[i] = nested_score.score(X,y)
        # train_scores[i] = cv_results['train_score'][i]
        # test_scores[i] = cv_results['test_score'][i]
       
    
    
    ### ========== TODO : END ========== ###
    
    return train_scores, test_scores


######################################################################
# main
######################################################################

def main() :
    np.random.seed(1234)
    
    # load data
    train_data = pd.read_csv("../data/phoneme_train.csv", header=None)
    label_col = train_data.columns[-1]
    X_train = train_data.drop([label_col], axis=1).to_numpy()
    y_train = train_data[label_col].to_numpy()
    
    ### ========== TODO : START ========== ###
    # part a: is data linearly separable?
    # hints: be sure to set parameters for Perceptron
    #        an easy parameter to miss is tol=None, a much stricter stopping criterion than default
    # professor's solution: 5 lines
    clf = Perceptron(tol=None, random_state=1234)
    clf.fit(X_train, y_train)
    print("Average accuracy is ", clf.score(X_train,y_train))
    
    ### ========== TODO : END ========== ###
    
    print()
    
    #========================================
    # part d: compare classifiers
    # it may take a few minutes to run since we are fitting several classifiers several times
    
    # setup
    ntrials = 10
    scores = []
    clf_strs = ["dummy", "perceptron", "logistic regression"]
    
    # nested CV to estimate performance
    for clf_str in clf_strs :
        clf, param_grid = get_classifier(clf_str)
        train_scores, test_scores = get_performance(clf, param_grid, X_train, y_train, ntrials)
        for i in range(ntrials):
            scores.append({'classifier': clf_str, 'fold': i,
                           'training': train_scores[i], 'testing': test_scores[i]})
        
        print(f"{clf_str}")
        print(f"\ttraining accuracy: {np.mean(train_scores):.3g} +/- {np.std(train_scores):.3g}")
        print(f"\ttest accuracy:     {np.mean(test_scores):.3g} +/- {np.std(test_scores):.3g}")
    print()
    
    # plot
    df = pd.DataFrame(scores)
    df = df.melt(id_vars=['classifier', 'fold'], var_name='dataset', value_name='accuracy')
    ax = sns.barplot(data=df, x="dataset", y="accuracy", hue='classifier', ci='sd')
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.3f}", xy=(p.get_x() + p.get_width() / 2., height),
                    xytext=(0, 3), textcoords='offset points', # 3 points vertical offset
                    ha='center', va='bottom')
    ax.set_title("Nested Cross-Validation Performance")
    ax.legend(title='classifier', loc=4)    # lower right
    plt.savefig("../plots/results.pdf")
    #plt.show()
    
    ### ========== TODO : START ========== ###
    # part e: compute significance using t-test
    # hints:
    # (1) given DataFrame
    #         df = pd.DataFrame({'prof': ['wu', 'wu', 'boerkoel'],
    #                            'class': ['cs151', 'cs158', 'cs151'],
    #                            'caps': [32, 20, 50]}
    #     that is,
    #         prof      class   cap
    #         0 wu        cs151   32
    #         1 wu        cs158   20
    #         2 boerkoel  cs151   50
    #     extract caps for Prof Wu's CS 158 classes via
    #         df[(df['prof'] == 'wu') & (df['class'] == 'cs158')]['cap']
    # (2) compute t-test via scipy.stats.ttest_rel(...)
    # professor's solution: 5 lines
    
    print("significance tests")
    
    
    
    ### ========== TODO : END ========== ###

if __name__ == "__main__" :
    main()