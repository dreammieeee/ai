import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import numpy as np

features, labels = make_blobs(n_samples=200, centers=[[2.0, 2.0], [3.0, 3.0]], 
                              cluster_std=0.75, n_features=2, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(features, labels)

def visualize_decision_boundary(data, target, classifier):
    step_size = 0.02  
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    
    grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max, step_size),
                                 np.arange(y_min, y_max, step_size))
    
    predictions = classifier.predict(np.c_[grid_x.ravel(), grid_y.ravel()])
    predictions = predictions.reshape(grid_x.shape)

    plt.contourf(grid_x, grid_y, predictions, alpha=0.6, cmap=plt.cm.coolwarm)
    
    scatter = plt.scatter(data[:, 0], data[:, 1], c=target, edgecolor='k', cmap=plt.cm.coolwarm)
    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary Visualization")
    plt.legend(handles=scatter.legend_elements()[0], labels=["Class A", "Class B"])
    plt.show()

visualize_decision_boundary(features, labels, log_reg)
