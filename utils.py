import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_classification(X_set, y_set, classifier, title, x_label, y_label):
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    colors = ('red', 'green')
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha=0.75, cmap=ListedColormap(colors))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for cat, color in zip(np.unique(y_set), colors):
        plt.scatter(X_set[y_set == cat, 0], X_set[y_set == cat, 1], c=color, label=cat)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()