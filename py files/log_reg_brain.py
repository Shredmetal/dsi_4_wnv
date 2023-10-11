from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

class LogRegBrain:
    """
    This class contains the key operations which each iteration of a combination of features will need to use.

    It creates a linear regression model based on the train and test data created by the DataCleaner class and saves
    the intercept, coefficient, rmse, train score, test score, and predictions and attributes.

    This class should have been implemented by inheriting from sk-learn's linear regression using __super__ but this
    was not done due to time constraints.
    """

    # Take in the X and y, instantiate sk-learn lr model and fit and save as attr
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.lr = LogisticRegression(solver='lbfgs', max_iter=10000)
        self.model = self.lr.fit(self.X_train, self.y_train)
        self.X_test = X_test
        self.y_test = y_test
        self.preds = self.model.predict(self.X_test)
        self.train_score = self.get_train_score()
        self.test_score = self.get_test_score()

    # returns r2 score of model based on train data
    def get_train_score(self):
        return self.model.score(self.X_train, self.y_train)

    # returns r2 score of model based on test data
    def get_test_score(self):
        return r2_score(self.y_test, self.preds)
