from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from softmax import *


class CrossValidation:
    def __init__(self, k, dataset):
        self.k = k
        self.d = dataset

    def validation_split(self, n):
        # get the number of data samples in each split
        n_val = n // self.k
        shuffled = np.random.permutation(n)
        for f in range(self.k):
            tr_inds = []
            # get the validation indexes
            val_inds = list(range(f * n_val, (f + 1) * n_val))
            # get the train indexes
            if f > 0:
                tr_inds = list(range(f * n_val))
            if f < self.k - 1:
                tr_inds = tr_inds + list(range((f + 1) * n_val, n))
            yield shuffled[tr_inds], shuffled[val_inds]

    def cross_validate_acc(self, model, optimizer=None):
        x = self.d["data"]
        y = self.d["target"]
        acc_valid = np.zeros(self.k)
        acc_train = np.zeros(self.k)
        i = 0
        iterations = 0
        for tr_inds, val_inds in self.validation_split(x.shape[0]):
            if optimizer is None:
                fit = model.fit(x[tr_inds], y[tr_inds])
                pred_valid = fit.predict(x[val_inds])
                pred_train = fit.predict(x[tr_inds])
            else:
                iterations, fit = model.fit(
                    x[tr_inds], y[tr_inds], optimizer, x[val_inds], y[val_inds]
                )
                pred_valid = fit.predict(x[val_inds]).argmax(axis=1)
                pred_train = fit.predict(x[tr_inds]).argmax(axis=1)
            acc_valid[i] = CrossValidation.validation_acc(y[val_inds], pred_valid)
            acc_train[i] = CrossValidation.validation_acc(y[tr_inds], pred_train)
            i += 1
        return iterations, acc_valid.mean(), acc_train.mean()

    @staticmethod
    def validation_acc(y, y_pred):
        return np.sum(y == y_pred) / y.shape[0]


class GridSearch:
    def __init__(self, k, dataset, batch, learning_rate, momentum, max_iters=1e3):
        # K fold CV
        self.k = k
        self.d = dataset
        self.batch = batch
        self.lr = learning_rate
        self.momentum = momentum
        self.max_iters = max_iters


    def parameters(self):
        for b in self.batch:
            for l in self.lr:
                for m in self.momentum:
                    yield b, l, m

    def accuracy_plot(self):
        err_valid = []
        param_valid = []

        err_train = []
        param_train = []
        count = 1
        cv = CrossValidation(self.k, self.d)
        for i, (b, l, m) in enumerate(self.parameters()):
            optimizer = GradientDescent(learning_rate=l, max_iters=1e4,
                                        epsilon=1e-8, batch_size=b, momentum=m)
            accuracy_valid = 0
            accuracy_train = 0
            for (tr_inds, val_inds) in cv.validation_split(len(self.d['data'])):
                model = SoftmaxRegression()
                model.fit(self.d['data'][tr_inds], self.d['target'][tr_inds], optimizer,
                          self.d['data'][val_inds], self.d['target'][val_inds])
                # validation accuracy
                pred_valid = np.argmax(model.predict(self.d['data'][val_inds]), axis=1)
                correct_valid = (pred_valid == self.d['target'][val_inds])
                accuracy_valid += correct_valid.sum() / correct_valid.size

                # training accuracy
                pred_train = np.argmax(model.predict(self.d['data'][tr_inds]), axis=1)
                correct_train = (pred_train == self.d['target'][tr_inds])
                accuracy_train += correct_train.sum() / correct_train.size

            err_valid.append((accuracy_valid / self.k) * 100)
            param_valid.append((b, l, m))

            err_train.append((accuracy_train / self.k) * 100)
            param_train.append((b, l, m))
            count += 1
        best_training_error = max(err_train)
        best_parameter_training = param_train[err_train.index(best_training_error)]

        best_validation_error = max(err_valid)
        best_parameter_validation = param_valid[err_valid.index(best_validation_error)]
        return ((best_training_error, best_parameter_training),
                (best_validation_error, best_parameter_validation))