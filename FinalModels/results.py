import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold




class ModelEvaluation:

    def classification_report(self, y_true, y_pred):
        """
        Prints the classification report.

        :param y_true: The ground truth target values.
        :param y_pred: The predicted target values.
        """
        report = classification_report(y_true, y_pred)
        print(report)

    def get_cm(self, y_true, y_pred):
        """
        Plots the confusion matrix.

        :param y_true: The ground truth target values.
        :param y_pred: The predicted target values.
        """
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["non-sarcastic", "sarcastic"])
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

    def get_roc(self, y_true, y_scores):
        """
        Plots the ROC curve and finds the best threshold.

        :param y_true: The ground truth target values.
        :param y_scores: The predicted probabilities of the positive class.
        :return: The best threshold value.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Calculate the G-mean for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))
        # Locate the index of the largest G-mean
        ix = np.argmax(gmeans)
        best_thresh = thresholds[ix]

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best Threshold (G-mean)')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()

        print(f'Best Threshold={best_thresh:.4f}')
        return best_thresh
    

    def cross_validation_score(self, model, X, y, scoring='accuracy'):
        """
        Performs 5-fold cross-validation and returns the scores for each fold.

        :param model: The machine learning model to be evaluated.
        :param X: The input features.
        :param y: The target labels.
        :param scoring: The scoring metric to use.
        :return: The cross-validation scores for each fold.
        """
        # Create a StratifiedKFold object to keep the same proportion of classes in each fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Perform cross-validation and return the scores
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        print(f"Cross-validation scores for each fold: {scores}")
        print(f"Mean score: {scores.mean()}")
        print(f"Standard Deviation of the scores: {scores.std()}")
        return scores
