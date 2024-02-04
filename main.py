from sklearn import datasets
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load our breast cancer dataset
data = datasets.load_breast_cancer()

# Store the feature data
X = data.data
# Store the target data
y = data.target

# Split the data using Scikit-learn's train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate the mean and variance for class 0 and class 1
mean_x0 = np.mean(X_train[y_train == 0], axis=0)
mean_x1 = np.mean(X_train[y_train == 1], axis=0)
var_x0 = np.var(X_train[y_train == 0], axis=0)
var_x1 = np.var(X_train[y_train == 1], axis=0)

# Parameters to use for the k-NN
param_inds = [22, 27, 23, 20, 13, 7, 0, 3, 2, 6]
X_train = X_train[:, param_inds]
X_test = X_test[:, param_inds]

print("Shape of X originally is: ", X.shape)

# Create and train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)

# Function to calculate accuracy
def calc_accuracy(predictions, labels):
    return np.mean(predictions == labels)

# Print accuracy
accuracy = calc_accuracy(predictions, y_test)
print(f"Accuracy levels are at: {accuracy}")

# Plotting the accuracy levels
knn_param_vec = [accuracy]  # You would fill this with actual accuracy values for different parameters

fontsize = 14
fontname = 'times new roman'

plt.plot(knn_param_vec, label='Number of Neighbors')
plt.xlabel('Number of Neighbors', fontname=fontname, fontsize=fontsize)
plt.ylabel('Accuracy', fontname=fontname, fontsize=fontsize)
plt.title('k-NN Accuracy', fontname=fontname, fontsize=fontsize)
plt.xticks(fontname=fontname, fontsize=fontsize)
plt.yticks(fontname=fontname, fontsize=fontsize)
plt.legend()
plt.show()
