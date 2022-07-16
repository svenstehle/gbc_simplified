# %%
# Copyright (c) <2022>, <Sven Stehle>
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pdb

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

rng = 42

n_classes = 4

X, y = make_classification(
    n_samples=500, n_features=12, n_informative=8, n_redundant=2, n_repeated=2, n_classes=n_classes, random_state=rng
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rng)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create the stump
Tree_model = DecisionTreeClassifier(criterion="entropy")
predictions = np.mean(cross_validate(Tree_model, X_train, y_train, cv=10)['test_score'])

print(f'The accuracy is: {predictions*100:.2f}%')

# %%

# heavily borrowed from scikit learn

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class GradientBoosting:
    def __init__(self, n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators = None
        self.predictions = None
        self.classes_ = None
        self.n_classes_ = None
        self.init_ = self._init_estimator()

    def fit(self, X: np.ndarray, y: np.ndarray):
        random_state = np.random.RandomState(self.random_state)
        self.init_.fit(X, y)
        raw_predictions = self._get_init_raw_predictions(X, self.init_)
        # get the classes
        self.classes_ = getattr(self.init_, "classes_", None)
        self.n_classes_ = len(self.classes_)    # TODO merge dummyestimator and fix this?

        # run the boosting by creating ``n_estimators`` x ``n_classes``
        self.estimators = np.empty((self.n_estimators, self.n_classes_), dtype=object)

        # compute the residuals for each class and each boost step
        for iboost in range(self.n_estimators):
            raw_predictions = self._fit_stage(iboost, X, y, raw_predictions, random_state)

        return self

    def _fit_stage(self, iboost: int, X: np.ndarray, y: np.ndarray, raw_predictions: np.ndarray, random_state: int):
        original_y = y
        # raw_predictions are in log-probability
        for k in range(self.n_classes_):
            # encode y as array of [0, 1]. 1 if element equals k, 0 otherwise
            y = np.where(original_y == k, 1, 0)

            # get the negative gradients (residuals) for each class
            residuals = self._negative_gradient(y, raw_predictions, k)

            # fit regression tree on residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, splitter="best", random_state=random_state)
            tree.fit(X, residuals)

            # update predictions
            raw_predictions[:, k] = self._decision_function(X, raw_predictions, k, tree)

            # add tree to ensemble
            self.estimators[iboost, k] = tree
        return raw_predictions

    def _negative_gradient(self, y: np.ndarray, raw_predictions: np.ndarray, k: int):
        """Compute negative gradients for class k.
        Negative gradients can be understood as the directional error and the magnitude of
        the necessary change in probability to correctly predict each respective label with
        regard to class k.

        Args:
            y (ndarray of shape (n_samples, )): Target labels
            raw_predictions (ndarray of shape (n_samples, K)): The raw predictions as log-probabilities
                of the tree ensemble at iteration ``i - 1`` for all ``K`` classes.
            k (int): index of the class

        Returns:
            ndarray of shape (n_samples, ): negative gradient for class k
        """
        log_probas_class_k = raw_predictions[:, k]
        probas_class_k = np.exp(log_probas_class_k)
        clipped_probas_class_k = np.nan_to_num(probas_class_k)
        return y - clipped_probas_class_k

    def _decision_function(self, X: np.ndarray, raw_predictions: np.ndarray, k: int, tree: DecisionTreeRegressor):
        # merge a tree's predictions (negative gradient) with the raw_predictions (log-probabilities)
        return raw_predictions[:, k].ravel() + self.lr * tree.predict(X)

    def _init_estimator(self):
        return DummyClassifier(strategy="prior")

    def _get_init_raw_predictions(self, X: np.ndarray, estimator: DummyClassifier):
        # initialize raw predictions as log-probabilities
        probas = estimator.predict_proba(X)
        eps = np.finfo(np.float32).eps
        probas = np.clip(probas, eps, 1 - eps)
        raw_predictions = np.log(probas).astype(np.float64)
        return raw_predictions

    def predict(self, X: np.ndarray):
        # make a prediction with each estimator
        classes = self.classes_[:, np.newaxis]
        raw_predictions = self._get_init_raw_predictions(X, self.init_)

        for i in range(self.n_estimators):
            for k in classes:
                # apply decision function to each class
                try:
                    raw_predictions[:, k] = self._decision_function(
                        X,
                        raw_predictions,
                        k,
                        self.estimators[i, k][0],
                    ).reshape(-1, 1)
                except ValueError:
                    pdb.set_trace()
        proba = self._raw_prediction_to_proba(raw_predictions)
        # get class label for max probability per row
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def _raw_prediction_to_proba(self, raw_predictions: np.ndarray):
        # convert logprob to probabillity
        return np.nan_to_num(np.exp(raw_predictions))


#%%
######Plot the accuracy of the model against the number of stump-estimators used##########

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

plt.style.use('fivethirtyeight')

number_of_base_learners = 60

fig = plt.figure(figsize=(8, 6))
ax0 = fig.add_subplot(111)
accuracies_custom = []
accuracies_sklearn = []
accuracies_sklearn_rf = []
learning_rate = 0.5
max_depth = 3

for i in range(1, number_of_base_learners + 1):
    # custom GBC
    model = GradientBoosting(
        n_estimators=i,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=rng,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies_custom.append(acc)

    # sklearn GBC
    model = GradientBoostingClassifier(
        n_estimators=i,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=rng,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies_sklearn.append(acc)

    # sklearn RF
    model = RandomForestClassifier(
        n_estimators=i,
        random_state=rng,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies_sklearn_rf.append(acc)

ax0.plot(range(len(accuracies_custom)), accuracies_custom, alpha=0.5)
ax0.plot(range(len(accuracies_sklearn)), accuracies_sklearn, alpha=0.5)
ax0.plot(range(len(accuracies_sklearn_rf)), accuracies_sklearn_rf, alpha=0.5)
plt.legend(['custom', 'sklearn_gbc', 'sklearn_rf'])
ax0.set_xlabel('# models used for Boosting ')
ax0.set_ylabel('accuracy')
print(
    f'With a number of {number_of_base_learners} base models we receive an accuracy of {accuracies_custom[-1]*100:.2f}%'
)

plt.show()

# %%

# investigate the negative gradient computation on raw predictions
k = 2
original_y = np.array([0, 1, 2, 2])

raw_predictions = np.log(np.array([
    [0.3, 0.5, 0.2],
    [0.3, 0.2, 0.5],
    [0.6, 0.1, 0.3],
    [0.2, 0.1, 0.7],
]))
raw_predictions
#%%

y = np.array(original_y == k, dtype=np.float64)
y, y.shape

#%%
# the log-probabilities for class k
log_prob = raw_predictions[:, k]
log_prob

#%%
# the probabilities for class k
probas = np.exp(log_prob)

#%%
# wrapper in case of nan
probas_clipped = np.nan_to_num(probas)
probas_clipped

#%%
# the negative gradients
y - probas_clipped

# %%
