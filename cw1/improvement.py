##############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train_and_predict() function. 
#             You are free to add any other methods as needed. 
##############################################################################

import numpy as np
from classification import *

class Prune:

    def prune_single(node, x, y, classifier):
        if node.left.leaf and node.right.leaf:
            if (len(x) == 0):
                return
            else:
                label, counts = np.unique(node.y, return_counts=True)
                after_label = label[np.argmax(counts)]
                prediction = classifier.predict(x)
                best_acc = Evaluation.accuracy(y,prediction)
                aft_acc = np.sum(node.y == after_label) / len(node.y)
                if aft_acc > best_acc:
                    node.leaf = True
                    node.left = None
                    node.right = None
                    node.attribute = -1
                    node.value = -1
                    node.label = after_label
                    return
                else:
                    return

        else:

            children_datasets = DecisionTreeClassifier.split_dataset(x,y,node)
            i = 0
            left_subset_valid_x = None
            left_subset_valid_y = None
            right_subset_valid_x = None
            right_subset_valid_y = None
            for child_dataset in children_datasets:
                if i == 0:
                    left_subset_valid_x = child_dataset[0]
                    left_subset_valid_y = child_dataset[1]
                    i = i + 1
                    continue
                if i == 1:
                    right_subset_valid_x = child_dataset[0]
                    right_subset_valid_y = child_dataset[1]

            if node.left.leaf == False and node.right.leaf == True:
                Prune.prune_single(node.left, left_subset_valid_x, left_subset_valid_y, classifier)
            elif node.left.leaf == True and node.right.leaf == False:
                Prune.prune_single(node.right, right_subset_valid_x, right_subset_valid_y, classifier)

            elif node.left.leaf == False and node.right.leaf == False:
                Prune.prune_single(node.left, left_subset_valid_x, left_subset_valid_y, classifier)
                Prune.prune_single(node.right, right_subset_valid_x, right_subset_valid_y, classifier)

                return

    def prune(features, labels, n_folds):
        """
        given features, labels, and number of folds
        prune automatically calculates the best tree structure using prune
        """
        accuracies = np.zeros((n_folds,))
        recalls = np.zeros((n_folds, ))
        precisions = np.zeros((n_folds, ))
        f1s = np.zeros((n_folds, ))
        nodes = []
        for i, (train_indices, test_indices) in enumerate(CrossValidation.train_test_k_fold(n_folds, len(labels))):
            x_train = features[train_indices, :]
            y_train = labels[train_indices]
            x_test = features[test_indices, :]
            y_test = labels[test_indices]
            classifier = DecisionTreeClassifier()
            node = classifier.fit(x_train, y_train)
            predictions = classifier.predict(x_test)
            accuracy1 = Evaluation.accuracy(y_test, predictions)
            Prune.prune_single(node, x_test, y_test, classifier)
            classifier = DecisionTreeClassifier(node)
            nodes.append(node)
            predictions = classifier.predict(x_test)
            accuracy2 = Evaluation.accuracy(y_test, predictions)
            accuracies[i] = max(accuracy2,accuracy1)
            recalls[i] = Evaluation.recall(y_test, predictions)[1]
            precisions[i] = Evaluation.precision(y_test, predictions)[1]
            f1s[i] = Evaluation.f1_score(y_test, predictions)[1]

        print(f"all accuracies in {n_folds} cross validation:")
        print(accuracies)
        print(f"mean: {accuracies.mean():.2f}")
        print(f"standard deviation: {accuracies.std():.2f}")
        
        print(f"all recalls in {n_folds} cross validation:")
        print(recalls)
        print(f"mean: {recalls.mean():.4f}")
        print(f"standard deviation: {recalls.std():.4f}")
        
        print(f"all precisions in {n_folds} cross validation:")
        print(precisions)
        print(f"mean: {precisions.mean():.4f}")
        print(f"standard deviation: {precisions.std():.4f}")
        
        print(f"all f1_score in {n_folds} cross validation:")
        print(f1s)
        print(f"mean: {f1s.mean():.4f}")
        print(f"standard deviation: {f1s.std():.4f}")
        return nodes, nodes[accuracies.argmax()]




# This is a single function, not with in class
def train_and_predict(x_train, y_train, x_test, x_val, y_val):
    """ Interface to train and test the new/improved decision tree.
    
    This function is an interface for training and testing the new/improved
    decision tree classifier. 

    x_train and y_train should be used to train your classifier, while 
    x_test should be used to test your classifier. 
    x_val and y_val may optionally be used as the validation dataset. 
    You can just ignore x_val and y_val if you do not need a validation dataset.

    Args:
    x_train (numpy.ndarray): Training instances, numpy array of shape (N, K) 
                       N is the number of instances
                       K is the number of attributes
    y_train (numpy.ndarray): Class labels, numpy array of shape (N, )
                       Each element in y is a str 
    x_test (numpy.ndarray): Test instances, numpy array of shape (M, K) 
                            M is the number of test instances
                            K is the number of attributes
    x_val (numpy.ndarray): Validation instances, numpy array of shape (L, K) 
                       L is the number of validation instances
                       K is the number of attributes
    y_val (numpy.ndarray): Class labels of validation set, numpy array of shape (L, )
    """

    best_node = Prune.prune(x_train,y_train,10)
    classifier = DecisionTreeClassifier(best_node)
    predictions = classifier.predict(x_test)
    return predictions
