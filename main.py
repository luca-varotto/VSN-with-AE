import math
import numpy as np
# from sklearn.utils.testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning
# from warnings import simplefilter
# simplefilter("ignore", category=ConvergenceWarning)

############################################################################################################################

from sim_manager import Manager

############################################################################################################################

# STIMULATION MODEL IDENTIFICATION OF A LARGE VSN VIA DEEP EMBEDDED FEATURES AND GGMs

# Compare stimulation model identification performance between GMM, V-GMM, GMM+AE and V-GMM+AE

############################################################################################################################

'''
'''
def create_events_pdf(M, uniform_flag=True):
  if uniform_flag:
    events_pdf = (1/M)*np.ones(M) 
  else:
    events_pdf = np.zeros(M)
    for m in range(M):
      events_pdf[m] = np.random.uniform(0.3,0.8)
    events_pdf /= sum(events_pdf)
  return events_pdf

############################################################################################################################

# PRE-TRAINED MODEL 
# Uncomment the following line to use a pre-trained model
# model_folder = '07022020-181834'

try: # if model model_folder is not defined, this section is skipped
  
  # instantiate an empty simulation manager 
  manager = Manager()

  # use the manager to load the pre-trained model, its scaler and all parameters used during its training
  manager.load(model_folder)

  # GENERATE MEASUREMENTS (DATASET)
  manager.user_params['n'] = 1000
  h, labels = manager.data_generator()

  # CREATE DEEP EMBEDDED FEATURES
  # train/test dataset split
  manager.user_params['split_ratio'] = 0.0 # only test data
  X_train, X_test, y_train, y_test = manager.ae.data_preproc(h,labels,manager.user_params['split_ratio']) 
  manager.dnn.scaler = manager.ae.scaler
  print("X_train shape: {} \nX_test shape: {} \ny_train shape: {} \ny_test shape: {} \n".format(\
      np.shape(X_train),np.shape(X_test),np.shape(y_train),np.shape(y_test)))

except NameError: # train a new model
  
  # SETUP PARAMETERS
  user_params = {
    
    # UNDERLYING STIMULATION MODEL
    'M': 3, # number of events (unknown)
    'M_max': 20, # a-priori knowledge on the upper bound on M
    'events': None, # events labels [defined later]
    'p_events': None, # events a-priori probability (unknwon) [defined later]
    'T': None, # underlying stimulation matrix (unknwon) [defined later]

    # VSN
    'N': 15, # number of nodes

    # DATASET
    'p_succ': 0.8,  # success probability for each node (success = correct measurement)
    'n': int(5*1.0E+3),  # number of measurements
    'confidence_range': [0.7,1.0], # confidence associated to a single measurement
    'split_ratio': 0.8, # train-test split ratio

    # AE
    'structure_array': None,  # encoder structure (decoder is symmetric) [defined later]
    'epochs': 30, 
    'batch_size': 30,
     
    # DNN
    'use_DNN': True # boolean flag to activate/deactivate the comparison of AE with a DNN
  }
  user_params['events'] = range(user_params['M'])
  user_params['p_events'] = create_events_pdf(user_params['M'],uniform_flag=False)
  user_params['structure_array']= [math.ceil(3*user_params['N']/4),math.ceil(user_params['N']/2),math.ceil(user_params['N']/4),2]

  # instantiate the simulation manager according to the user-defined parameters
  manager = Manager(user_params=user_params)

  # define the triggering model through the stimulation matrix (supposed injective)
  manager.user_params['T'] = manager.create_T()

  # GENERATE MEASUREMENTS (DATASET)
  h, labels = manager.data_generator()

  # TRAINING
  # train/test dataset split
  X_train, X_test, y_train, y_test = manager.ae.data_preproc(h,labels,manager.user_params['split_ratio']) 
  if manager.user_params['use_DNN']:
    manager.dnn.scaler = manager.ae.scaler
  print("X_train shape: {} \nX_test shape: {} \ny_train shape: {} \ny_test shape: {} \n".format(\
        np.shape(X_train),np.shape(X_test),np.shape(y_train),np.shape(y_test)))
  # AE training
  history = manager.ae.AEtrain(X_train, X_test, manager.user_params['epochs'], manager.user_params['batch_size'])  
  manager.ae.plot_history(history) # plot metrics
  # DNN training
  if manager.user_params['use_DNN']:
    history = manager.dnn.AEtrain(X_train, X_test, manager.user_params['epochs'], manager.user_params['batch_size'])  
    manager.dnn.plot_history(history) # plot metrics

# dictionary of boolean flags related to clustering plots
want_to_plot_dict = {
  'plot_BIC': True,
  'plot_prediction': True,
  'plot_cm': False
}  
# CLUSTERING ON RAW MEASUREMENTS
# GMM  
acc_gmm,labels_gmm = manager.clustering(h,labels,'GMM',want_to_plot_dict)
M_est_list = [np.array(labels_gmm).max()+1]
acc_list = [acc_gmm]
# V-GMM 
acc_vgmm,labels_vgmm = manager.clustering(h,labels,'V-GMM',want_to_plot_dict) 
M_est_list.append(np.array(labels_vgmm).max()+1)
acc_list.append(acc_vgmm)

# CLUSTERING ON EMBEDDED FEATURES
# generate deep embedded features on test data
encoded, decoded,reconstructed_data = manager.ae.AEpredict(X_test)
# GMM 
acc_gmm_DEC,labels_gmm_DEC = manager.clustering(encoded, y_test,'GMM',want_to_plot_dict)
M_est_list.append(np.array(labels_gmm_DEC).max()+1)
acc_list.append(acc_gmm_DEC)
# V-GMM 
acc_vgmm_DEC,labels_vgmm_DEC = manager.clustering(encoded, y_test,'V-GMM',want_to_plot_dict)
M_est_list.append(np.array(labels_vgmm_DEC).max()+1)
acc_list.append(acc_vgmm_DEC)

# CLUSTERING ON TRANSFORMED FEATURES
if manager.user_params['use_DNN']:
  # generate deep transformed features on test data
  encoded_dnn, decoded_dnn,reconstructed_data_dnn = manager.dnn.AEpredict(X_test)
  # GMM 
  acc_gmm_DNN,labels_gmm_DNN = manager.clustering(encoded_dnn, y_test,'GMM',want_to_plot_dict)
  M_est_list.append(np.array(labels_gmm_DNN).max()+1)
  acc_list.append(acc_gmm_DNN)
  # V-GMM 
  acc_vgmm_DNN,labels_vgmm_DNN = manager.clustering(encoded_dnn, y_test,'V-GMM',want_to_plot_dict)
  M_est_list.append(np.array(labels_vgmm_DNN).max()+1)
  acc_list.append(acc_vgmm_DNN)

# if manager.user_params['split_ratio'] > 0.0: # save only in case of previous training 
#   # save the model
#   user_choice = input('\nDo you want to save the model? \nPress 1 to say YES, any other key otherwise: ') 
#   manager.save(user_choice)

            # ***** RESULTS ANALYSIS *****

# CLUSTERING PERFORMANCE
# bar chart of components weights 
manager.plotter.weights_bars(manager.gmm_models_list)
string_acc = ''
string_M_est = ''
for i in range(len(manager.gmm_models_list)):
  string_acc += '\n' + manager.gmm_models_names[i]  + ' = ' + str(acc_list[i])
  string_M_est += '\n' + manager.gmm_models_names[i] + ' = ' + str(M_est_list[i])
print("\n       CLUSTERING RESULTS: \
      \n\nAccuracy",\
      string_acc,\
      "\n\nEstimated number of components",\
      string_M_est)

# REPROJECTION ONTO THE UNIT HYPERCUBE 
# map each cluster centroid to the closest hypercube vertex and discover how many EFFECTIVE clusters are
# detected and how many of these coincide with the TRUE ones (reconstruction accuracy)
trueVertices_counter_list, err_list, vertices_weights_dict_list, KL_list = manager.reprojection()
# compute number of effective componentss (for each gmm model)
M_eff_list = [len(list(vertices_weights_dict_list[i].values())) for i in range(len(manager.gmm_models_list)) ] 
string_correct_assignments = ''
string_err = ''
string_M_eff = ''
string_KL = ''
for i in range(len(manager.gmm_models_list)):
  string_correct_assignments += '\n' + manager.gmm_models_names[i]  + ' = ' + str(trueVertices_counter_list[i])
  string_err += '\n' + manager.gmm_models_names[i] + ' = ' + str(err_list[i])
  string_M_eff += '\n' + manager.gmm_models_names[i] + ' = ' + str(M_eff_list[i])
  string_KL += '\n' + manager.gmm_models_names[i] + ' = ' + str(KL_list[i])
print("\n       REPROJECTION RESULTS: \
      \nNumber of correct assignments",\
      string_correct_assignments,
      "\n\nReconstruction error", \
      string_err,
      "\n\nNumber of effective components",\
      string_M_eff,
      "\n\nKL Divergence",
      string_KL
      )
# bar chart of effective components weights 
manager.plotter.weights_bars(manager.gmm_models_list,vertices_weights_dict_list)

