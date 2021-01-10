from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import seaborn as sns
from tqdm import tqdm
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

############################################################################################################################

import clustering_plotting_tools

############################################################################################################################

# Gaussian Mixture Model object
class GMM:

  """ Constructor """
  def __init__(self, gmm_model=None):
    self.gmm_model = gmm_model 
    # if gmm_model is not None:
    #   self.means = gmm_model.means_ + np.zeros(np.shape(gmm.means_))
    #   self.covariances = gmm_model.covariances_ + np.zeros(np.shape(gmm.covariances_))
    #   self.weights = gmm_model.weights_ + np.zeros(np.shape(gmm.weights_))
    #   self.M = np.shape(gmm_model.means_)[0]
    #   self.log_likelihoods = []

  """ Generate a GMM model by choosing the optimal number of components 
  through the BIC criterion 
  X: dataset nxp (n= # datapoints, p= # features)
  M_range: range in which BIC is applied to choose the number of components
  type = 'standard' or 'bayesian' GMM
  want_to_plot: boolean if the BIC must be plotted
  """
  def GMM_generation(self, X, M_range, type = 'standard',weight_concentration_prior_type='dirichlet_distribution',\
     weight_concentration_prior=1.0E-3,want_to_plot=False):
    if type == 'standard': 
      # choose the best number of components for the GMM according to the BIC criterion
      models = [GaussianMixture(n, covariance_type='full', init_params='kmeans',\
        max_iter=300,n_init=1).fit(X) for n in M_range] # fit through EM algotithm
      BIC_scores = [m.bic(X) for m in models] # compute BIC scores
      best_model_idx = np.argmin(BIC_scores) # minimize BIC scores
      gmm = models[best_model_idx]

      # plot BIC criterion
      if want_to_plot:
        fig = plt.figure(figsize=(10,5))
        ax = plt.gca()
        plt.plot(M_range, BIC_scores, linestyle='-', marker='o') 
        # plt.plot(n_components, [m.aic(X) for m in models], label='AIC', linestyle='-', marker='o')
        ax.scatter(M_range[best_model_idx], BIC_scores[best_model_idx], s=100, c = 'r') 
        plt.xlabel(r'$M$')
        plt.ylabel(r'$J_{BIC}(M)$')
        plt.title('BIC')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=len(models)+1,integer=True))
        plt.show()

    else:
      # Fit a Dirichlet process Gaussian mixture
      gmm = BayesianGaussianMixture(n_components=M_range[-1],covariance_type='full', \
        init_params='kmeans',weight_concentration_prior_type=weight_concentration_prior_type,\
          weight_concentration_prior=weight_concentration_prior, max_iter=300,n_init=1).fit(X)

    return gmm

  """ Use the GMM for prediction
  X: input data
  probs: where to store the probability of belonging to the predicted cluster
  labels: where to store the predicted labels
  want_to_plot: boolean if the predictions must be plotted
  """
  def prediction(self,X,probs=[], labels= [],want_to_plot=False):

    # compute datapoints responsibilities
    if len(probs)==0:
        probs = self.gmm_model.predict_proba(X)

    # compute datapoints label according to the max responsibility
    if len(labels) ==0:
        labels = self.gmm_model.predict(X)

    # N = np.shape(X)[0]
    # probs = np.zeros((N,self.M))
    # labels = list()
    # for i in range(N):
    # for mu_k,cov_k,p_k,k in zip(self.means,self.covariances,self.weights,range(self.M)):
    # mn = multivariate_normal(mean=mu_k,cov=cov_k)
    # probs[i,k] = p_k*mn.pdf(X[i,:])/np.sum([pi_c*multivariate_normal(mean=mu_c,cov=cov_c).pdf(X[i,:]) \
    # for pi_c,mu_c,cov_c in zip(self.weights,self.means,self.covariances+1e-6*np.identity(2))],axis=0) 
    # labels.append(np.argmax(probs[i,:]))

    if want_to_plot:
      scale = 20 * probs.max(1)**2
      clustering_plotting_tools.plot_gmm(self.gmm_model.means_, self.gmm_model.covariances_, self.gmm_model.weights_, X, scale, labels)
      
    return labels
    
  ''' Compute confusion matrix
  labels_true: groundtruth
  labels_predicted: clustering result
  want_to_plot: boolean if the confusion matrix must be showed
  '''
  def confusion_matrix(self, labels_true, labels_predicted, want_to_plot=False):
    # compute confusion matrix
    # cm = confusion_matrix(labels_true, labels_predicted)

    # compute accuracy
    labels_predicted = np.array(labels_predicted)
    y_true = np.array(labels_true).astype(np.int64)
    D = max(labels_predicted.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # Confusion matrix.
    for i in range(labels_predicted.size):
        w[y_true[i],labels_predicted[i]] += 1
    ind = linear_assignment(-w)
    acc = sum([w[i, j] for i, j in ind]) / labels_predicted.size
    
    if want_to_plot:
      plt.figure(figsize=(5, 5))
      sns.heatmap(w, annot=True, fmt="d")
      plt.title("Confusion matrix", fontsize=30)
      plt.ylabel('True label', fontsize=25)
      plt.xlabel('Clustering label', fontsize=25)
      plt.show()
    return w, acc, ind

############################################################################################################################

class kMeans:

  """ Constructor """
  def __init__(self, kmeans_model=None):
    self.kmeans_model = kmeans_model 

  """ Generate a kmeans model by choosing the optimal number of clusters 
  through the elbow methods 
  X: dataset nxp (n= # datapoints, p= # features)
  K_range: range in which elbow method is applied to choose the number of clusters
  want_to_plot: boolean if the elbow method must be plotted
  """
  def kmeans_generation(self, X, K_range, want_to_plot=False):
    # choose the best number of clusters 
    Sum_of_squared_distances = []
    models = []
    for k in K_range:
      km = KMeans(n_clusters=k,random_state=1)
      km = km.fit(X)
      models.append(km)
      Sum_of_squared_distances.append(km.inertia_)

    # plot elbow method
    if want_to_plot:
      fig = plt.figure(figsize=(10,6))
      ax = plt.gca()
      plt.plot(K_range, Sum_of_squared_distances, label='', linestyle='-', marker='o') 
      plt.xlabel(r'$k$')
      plt.ylabel(r'Sum_of_squared_distances')
      plt.title('Elbow Method For Optimal k')
      ax.xaxis.set_major_locator(MaxNLocator(nbins=len(models)+1,integer=True))
      plt.show()

    return models

  """ Use the kmeans for prediction
  X: input data
  want_to_plot: boolean if the predictions must be plotted
  """
  def prediction(self,X,want_to_plot=False):

    # compute labels 
    labels = self.kmeans_model.predict(X)

    num_features = np.shape(X)[1]
    if want_to_plot and num_features == 2:
      clustering_plotting_tools.plot_kmeans(self.kmeans_model.cluster_centers_, X, labels)

    return labels
    
  ''' Compute confusion matrix
  labels_true: groundtruth
  labels_predicted: clustering result
  want_to_plot: boolean if the confusion matrix must be showed
  '''
  def confusion_matrix(self, labels_true, labels_predicted, want_to_plot=False):
    # compute confusion matrix
    cm = confusion_matrix(labels_true, labels_predicted)

    # compute accuracy
    labels_predicted = np.array(labels_predicted)
    y_true = np.array(labels_true).astype(np.int64)
    D = max(labels_predicted.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # Confusion matrix.
    for i in range(labels_predicted.size):
        w[labels_predicted[i], y_true[i]] += 1
    ind = linear_assignment(-w)
    acc = sum([w[i, j] for i, j in ind])  / labels_predicted.size

    if want_to_plot:
      plt.figure(figsize=(10, 10))
      sns.heatmap(cm, annot=True, fmt="d")
      plt.title("Confusion matrix", fontsize=30)
      plt.ylabel('True label', fontsize=25)
      plt.xlabel('Clustering label', fontsize=25)
      plt.show()
    return cm, acc, ind

