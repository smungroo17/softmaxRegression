import numpy as np
from collections import deque


class SoftmaxRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        self.X = None
        self.y = None
        self.w = None

    def fit(self, x, y, optimizer, x_validation, y_validation):
        N, D = x.shape
        num_classes = np.max(y) + 1
        if self.add_bias:
            x = np.column_stack([np.ones(N), x])
            x_validation = np.column_stack(
                [np.ones(x_validation.shape[0]), x_validation]
            )
            D += 1
        # initialize the weights
        w0 = np.zeros((D, num_classes))
        # fit the model
        self.X = x
        self.y = one_hot_encoding(y, num_classes)
        iter, self.w = optimizer.run(self.X, self.y, w0, x_validation, y_validation)
        return iter, self

    def predict(self, x):
        if self.add_bias:
            x = np.column_stack([np.ones(x.shape[0]), x])
        x_trans = np.dot(x, self.w)
        return softmax(x_trans)


class GradientDescent:
    def __init__(self, learning_rate=0.001, max_iters=1e4, epsilon=1e-8, batch_size=30, momentum=0.9):

        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.momentum = momentum

    def run(self, x, y, w, x_validation, y_validation):
        grad = np.inf
        prev_grad = None
        t = 0
        best_acc = 0  # best validation accuracy seen so far
        non_improving_iters = 0  # number of iters where best_acc doesn't improve
        non_improving_iters_max = 40
        prev_models = deque(maxlen=non_improving_iters_max)
        prev_accs = deque(maxlen=non_improving_iters_max)
        while (
                np.linalg.norm(grad) > self.epsilon
                and t < self.max_iters
                and non_improving_iters < non_improving_iters_max
        ):
            # calculate new w using minibatch
            batch_indices = np.random.randint(x.shape[0], size=self.batch_size)
            grad = gradient(x[batch_indices, :], y[batch_indices, :], w)
            if prev_grad is not None:
                grad = prev_grad * self.momentum + grad * (1 - self.momentum)
            prev_grad = grad
            w = w - self.learning_rate * grad
            t += 1
            # calculate validation error
            pred = np.argmax(
                softmax(np.dot(x_validation, w)), axis=x_validation.ndim - 1
            )
            acc = np.sum(pred == y_validation) / pred.shape[0]
            # conditionally save the model: if there are less than
            # non_improving_iters_max then save. If this iteration's accuracy is worse
            # than the best seen so far, then increment a counter. If this iteration's
            # accuracy is the best seen so far then save. Else
            best_acc = max(best_acc, acc)
            if len(prev_models) < non_improving_iters_max:
                prev_models.append(w)
                prev_accs.append(acc)
            elif acc < best_acc:
                non_improving_iters += 1
            else:
                non_improving_iters = 0
                prev_models.append(w)
                prev_accs.append(acc)
        return t, prev_models[int(np.argmax(prev_accs))]


def gradient(x, y, w):
    yh = softmax(np.dot(x, w))
    new_w = np.dot(x.T, yh - y) / x.shape[0]
    return new_w


def softmax(x):
    exp = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
    return exp / np.sum(exp, axis=1)[:, np.newaxis]


def one_hot_encoding(x, num_classes):
    return (np.arange(num_classes) == x[:, None]).astype(float)


def string_to_numerical_categories(x):
    empty = []
    to_return = []
    dict = {}
    for e in x:
        if e not in empty:
            empty.append(e)
    num_classes = len(empty)
    for i, e in enumerate(empty):
        dict[e] = i

    for e in x:
        to_return.append(dict[e])

    return np.array(to_return)

