from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


class LogRegBrain:
    """
    This class contains the key operations which each iteration of a combination of features will need to use.

    It creates a logistic regression model based on the train and test data created by the DataCleaner class and saves
    the ROC AUC score and predictions.

    In fitting the model, it also checks for the best penalty to apply and handles class imbalances using SMOTE.
    """

    # Take in the X and y, instantiate sk-learn lr model and fit and save as attr
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.pipe_lr = Pipeline(steps=[
            ('sampling', SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))),
            ('lr', LogisticRegression())
        ])
        self.pipe_lr_params = {
            'sampling__sampling_strategy': [1.0],
            'sampling__random_state': [42],
            'lr__C': [1],
            'lr__max_iter': [10000],
            'lr__penalty': ['l1', 'l2', None],
            'lr__random_state': [42],
            'lr__solver': ['saga'],
        }
        self.pipe_lr = GridSearchCV(self.pipe_lr, self.pipe_lr_params, scoring='roc_auc', cv=5, n_jobs=-1)
        self.pipe_lr.fit(self.X_train, self.y_train)
        self.X_test = X_test
        self.y_test = y_test
        self.preds = self.pipe_lr.predict(self.X_test)
        self.train_roc = self.get_train_roc()
        self.test_roc = self.get_test_roc()

    # returns roc-auc on train

    def get_train_roc(self):
        return self.pipe_lr.score(self.X_train, self.y_train)

    # returns roc-auc on test

    def get_test_roc(self):
        return self.pipe_lr.score(self.X_test, self.y_test)