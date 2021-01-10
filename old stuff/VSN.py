import numpy as np
import random as rnd
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, UpSampling2D, Activation
from keras import backend as K
# from keras.datasets import mnist

# number of nodes
n = 28
# set of nodes 
V = np.linspace(0,n-1,n, dtype=int)
np.random.shuffle(V) # shuffle to break symmetry in the following images generation
# number of clusters 
c = 4
# separate nodes in clusters
clusters_list = np.array_split(V,c)
# consider that each cluster is devoted to the detection of a specific event type;
# hence we have c event types (i.e., E0, E2, ... Ec). 
# Given event e_k belonging to the type E_k, the probability of node i to detect e_k is modeled as a Bernoulli process
# with success probability p_ik larger if c_i=k (c_i cluster of the i-th node).  
p_matrix = np.zeros((n,c))
p_succ = 0.8
for i in range(n):
    for k in range(c):
        p_matrix[i,k] = p_succ if i in clusters_list[k] else 1-p_succ # more sophisticated models can be used!!!!

# An events map is generated as
# A_ij = 1 iif nodes i and j both detect e_k
n_e = 10000 # number of events (i.e., dimension of the dataset)
dataset = []
y = []
for event_numer in tqdm(range(n_e)):
    e_k = rnd.randint(0,c-1)
    detections = []
    for i in range(n):
        D = np.random.binomial(1,p_matrix[i,e_k])
        detections.append(D)
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if detections[i] and detections[j]:
                A[i,j] = 1
    dataset.append(A.flatten())
    y.append(e_k)
    # plt.figure(figsize=(5,5))
    # plt.title("Event " + str(e_k))
    # plt.imshow(A)
    # plt.gray()
    # plt.show()

y = np.array(y)
dataset = np.array(dataset)
print("Dataset dimension:",np.shape(dataset))

# k-means
# dataset.reshape((dataset.shape[0], -1))
# kmeans = KMeans(n_clusters=c, random_state=1)
# y_pred_kmeans = kmeans.fit_predict(dataset)
# print("Accuracy kmeans: ", accuracy_score(y,y_pred_kmeans))

# #
# plt.figure(figsize=(5,5))
# plt.imshow(np.array(kmeans.labels_).reshape(-1,1))
# plt.gray()
# plt.figure(figsize=(5,5))
# plt.imshow(np.array(y).reshape(-1,1))
# plt.gray()

# # confusion matrix
# cm = confusion_matrix(y, y_pred_kmeans)
# plt.figure(figsize=(10, 10))
# sns.heatmap(cm, annot=True, fmt="d")
# plt.title("Confusion matrix", fontsize=30)
# plt.ylabel('True label', fontsize=25)
# plt.xlabel('Clustering label', fontsize=25)
# plt.show()

X = pd.DataFrame(dataset)
y = pd.Series(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=123)

# Reshape and Rescale the images
X_train = X_train.values.reshape(-1,n,n,1) #/ 255
X_test = X_test.values.reshape(-1,n,n,1) #/ 255
X_validate = X_validate.values.reshape(-1,n,n,1) #/ 255


# Build the autoencoder
model = Sequential()
model.add(Conv2D(14, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(14, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(1, kernel_size=3, padding='same', activation='relu'))

model.compile(optimizer='adam', loss="mse")
model.summary()

# Train the model
model.fit(X_train, X_train, epochs=3, batch_size=64, validation_data=(X_validate, X_validate), verbose=1)

# Fitting testing dataset
restored_testing_dataset = model.predict(X_test)

# Observe the reconstructed image quality
plt.figure(figsize=(20,5))
for i in range(c):
    index = y_test.tolist().index(i)
    plt.subplot(2, 10, i+1)
    plt.imshow(X_test[index].reshape((28,28)))
    plt.gray()
    plt.subplot(2, 10, i+11)
    plt.imshow(restored_testing_dataset[index].reshape((28,28)))
    plt.gray()

# Extract the encoder
encoder = K.function([model.layers[0].input], [model.layers[4].output])

# Encode the training set
encoded_images = encoder([X_test])[0].reshape(-1,7*7*7)

# Cluster the training set
kmeans = KMeans(n_clusters=c)
clustered_training_set = kmeans.fit_predict(encoded_images)

# Observe and compare clustering result with actual label using confusion matrix
cm = confusion_matrix(y_test, clustered_training_set)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()

# Plot the actual pictures grouped by clustering
fig = plt.figure(figsize=(20,20))
for r in range(c):
    cluster = cm[r].argmax()
    for c, val in enumerate(X_test[clustered_training_set == cluster][0:10]):
        fig.add_subplot(10, 10, 10*r+c+1)
        plt.imshow(val.reshape((28,28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('cluster: '+str(cluster))
        plt.ylabel('digit: '+str(r))