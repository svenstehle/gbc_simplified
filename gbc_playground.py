# %%
# Copyright (c) 2022, Sven Stehle
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

rng = 42

n_classes = 4

X, y = make_classification(
    n_samples=500,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    n_repeated=2,
    n_classes=n_classes,
    random_state=rng,
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

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingCustom:
    def __init__(self, n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42):
        """Constructor of GradientBoostingCustom. During init, self.base_estimator will be
        set as an instance of DummyClassifier. During the call to ``fit()``, self.base_estimator
        will become the root of the gradient boosting ensemble. It is the base for the first
        ``outputs``, which are log-probabilities for each of the K classes. Based on these
        ``outputs``, we build the ensemble with each new boosting stage and further refine
        the ``outputs``.

        Args:
            n_estimators (int, optional): the number of boosting stages. Defaults to 30.
            learning_rate (float, optional): the influence of each new estimator ``i`` on the current
                ``outputs`` of stage ``i-1``. Defaults to 0.1.
            max_depth (int, optional): the maximum depth of each DecisionTreeRegressor.
                Should be tuned. Note: we do not have to work with stumps here. Defaults to 3.
            random_state (int, optional): the seed for the random number
              generator. Defaults to 42.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.ensemble = None
        self.classes_ = None
        self.n_classes_ = None
        self.base_estimator = DummyClassifier(strategy="prior")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the algorithm on the input features ``X``
        and the Target labels ``y``.

        Args:
            X (np.ndarray of shape (n_samples, K)): input features.
            y (np.ndarray of shape (n_samples, )): Target labels.

        Returns:
            GradientBoostingCustom: instance of self.
        """
        random_state = np.random.RandomState(self.random_state)

        # get fit base estimator and initial outputs
        self.base_estimator.fit(X, y)
        outputs = self._get_initial_outputs(X, self.base_estimator)

        # get the class information
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # create an array that holds ``n_estimators`` for each ``n_classes`` ("ensemble of trees")
        self.ensemble = np.empty((self.n_estimators, self.n_classes_), dtype=object)

        # compute the residuals for each boost stage and class
        for iboost in range(self.n_estimators):
            for k in range(self.n_classes_):
                # get the negative gradient (residuals) for each class
                residuals = self._compute_residuals(y, outputs, k)

                # fit regression tree on residuals
                tree = DecisionTreeRegressor(max_depth=self.max_depth, splitter="best", random_state=random_state)
                tree.fit(X, residuals)

                # update predictions for class k with current regression tree
                outputs[:, k] = self._update_outputs(X, outputs, k, tree)

                # add tree to ensemble
                self.ensemble[iboost, k] = tree

        return self

    def _compute_residuals(self, y: np.ndarray, outputs: np.ndarray, k: int):
        """Compute residuals for class k.
        Residuals can be understood as negative gradient.
        They are expressed as the directional error
        and the magnitude of the necessary change in probability
        to correctly predict each respective label with
        regard to class k.

        Args:
            y (np.ndarray of shape (n_samples, )): Target labels.
            outputs (np.ndarray of shape (n_samples, K)): The outputs as
                log-probabilities of the tree ensemble at
                iteration ``i - 1`` for all ``K`` classes.
            k (int): index of the class.

        Returns:
            np.ndarray of shape (n_samples, ): negative gradient for class k.
        """
        # encode y as array of [0, 1]. 1 if element equals k, 0 otherwise
        y = np.where(y == k, 1, 0)

        # get outputs for class k and convert to clipped probabilities
        log_probas_class_k = outputs[:, k]
        probas_class_k = np.exp(log_probas_class_k)
        clipped_probas_class_k = np.nan_to_num(probas_class_k)

        # compute and return residuals (negative gradient)
        negative_gradient = y - clipped_probas_class_k
        return negative_gradient

    def _update_outputs(
        self,
        X: np.ndarray,
        outputs: np.ndarray,
        k: int,
        tree: DecisionTreeRegressor,
    ):
        """Updates the outputs with the predictions for the residuals
        of the current boosting stage.

        Args:
            X (np.ndarray of shape (n_samples, n_features)): input features.
            outputs (np.ndarray of shape (n_samples, K)): The outputs as
                log-probabilities of the tree ensemble at
                iteration ``i - 1`` for all ``K`` classes.
            k (int): index of the class.
            tree (DecisionTreeRegressor): the tree that was fit during the current stage.

        Returns:
            np.ndarray of shape (n_samples, ): the updated outputs for class k.
        """
        return outputs[:, k].ravel() + self.learning_rate * tree.predict(X)

    def _get_initial_outputs(self, X: np.ndarray, estimator: DummyClassifier):
        """Returns the initial outputs: DummyClassifier ``predict_proba`` always
        returns the empirical class distribution of y also known as the
        empirical class prior distribution. These probabilities are then clipped
        and returned as log-probabilities.

        Args:
            X (np.ndarray of shape (n_samples, n_features)): input features.
            estimator (DummyClassifier): the DummyClassifier object that will
                be fit during this function call.

        Returns:
            np.ndarray of shape (n_samples, K): The outputs as
                log-probabilities of the tree ensemble at
                iteration 0 for all ``K`` classes.
        """
        # initialize outputs as log-probabilities
        probas = estimator.predict_proba(X)
        eps = np.finfo(np.float32).eps
        probas = np.clip(probas, eps, 1 - eps)
        outputs = np.log(probas).astype(np.float64)
        return outputs

    def predict(self, X: np.ndarray):
        """Returns the predictions for input features X.

        Args:
            X (np.ndarray of shape (n_samples, n_features)): input features.

        Returns:
            np.ndarray of shape (n_samples, ): the predictions for X.
        """
        classes = self.classes_[:, np.newaxis]
        outputs = self._get_initial_outputs(X, self.base_estimator)

        # make predictions for each of the k classes with the respective estimators
        for i in range(self.n_estimators):
            for k in classes:
                # apply decision function to each class
                outputs[:, k] = self._update_outputs(
                    X,
                    outputs,
                    k,
                    self.ensemble[i, k][0],
                ).reshape(-1, 1)

        # convert log-probabilities to probability
        proba = np.exp(outputs)
        # get class label for max probability per row
        return np.argmax(proba, axis=1)


#%%
######Plot the accuracy of the model against the number of weak learners used##########

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'none'

number_of_base_learners = 50

fig = plt.figure(figsize=(8, 6))
fig.patch.set_facecolor('white')
ax0 = fig.add_subplot(111)
accuracies_custom = []
accuracies_sklearn = []
accuracies_sklearn_rf = []
learning_rate = 0.5
max_depth = 3

for i in range(1, number_of_base_learners + 1):
    # custom GBC
    model = GradientBoostingCustom(
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
    f'With a number of {number_of_base_learners} estimators '
    f'(boosting stages) we receive an accuracy of {accuracies_custom[-1]*100:.2f}%'
)

plt.show()

# %%
