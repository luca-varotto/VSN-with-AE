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

# Monte Carlo (MC) experiment to compare stimulation model identification performance between GMM, V-GMM, GMM+AE and V-GMM+AE

'''
'''
def create_events_pdf(M, uniform_flag=True):
  if uniform_flag:
    events_pdf = (1/M)*np.ones(M) 
  else:
    events_pdf = np.zeros(M)
    for m in range(M):
      events_pdf[m] = np.random.uniform(0.0,1.0)
    events_pdf /= sum(events_pdf)
  return events_pdf

############################################################################################################################

# Uncomment the following line to plot results of a saved MC experiment
# results_folder = '08162020-132733'

try: # if model results_folder is not defined, this section is skipped
  
  # instantiate an empty simulation manager 
  manager = Manager()

  # use the manager to load the all parameters used during MC experiment and the results
  results_dict = manager.load_MC(results_folder)

except NameError: # new MC experiment
  # SETUP PARAMETERS
  user_params = {
    
    # UNDERLYING STIMULATION MODEL
    'M': 3, # number of events (unknown)
    'M_max': 20, # a-priori knowledge on the upper bound on M
    'events': None, # events labels [defined later]
    'p_events': None, # events a-priori probability (unknwon) [defined later]
    'T': None, # underlying stimulation matrix (unknwon) [defined later]

    # VSN
    'N': 3, # number of nodes

    # DATASET
    'p_succ': 1.0,  # success probability for each node (success = correct measurement)
    'n': int(1.0E+4),  # number of measurements
    'confidence_range': [0.99,1.0], # confidence associated to a single measurement
    'split_ratio': 0.8, # train-test split ratio

    # AE
    'structure_array': None,  # encoder structure (decoder is symmetric) [defined later]
    'epochs': 30, 
    'batch_size': 30,

    # DNN
    'use_DNN': True # boolean flag to activate/deactivate the comparison of AE with a DNN
  }
  user_params['events'] = range(user_params['M'])
  user_params['p_events'] =  create_events_pdf(user_params['M'],uniform_flag=True)
  user_params['structure_array'] = [3,2]#[math.ceil(3*user_params['N']/4),math.ceil(user_params['N']/2),math.ceil(user_params['N']/4),2]
  # number of tests composing the MC experiment
  nb_tests = 10

  # STORAGE BUFFERS
  # some storage for GMM
  M_gmm = np.zeros(user_params['M_max'], dtype=int) # m-th entry counts the number of tests in which m components are estimated
  M_eff_gmm = np.zeros(user_params['M_max'], dtype=int) # m-th entry counts the number of tests in which m effective components are estimated
  acc_gmm_MC = [] # store accuracies
  weights_gmm = np.zeros((user_params['M_max'],nb_tests)) # store components weights at each test
  weights_eff_gmm = np.zeros((user_params['M_max'],nb_tests)) # store effective components weights at each test
  err_gmm = [] # store reprojection errors
  KL_gmm = [] # store KL divergence of the true events pdf w.r.t the estimated one
  # some storage for V-GMM
  M_vgmm = np.zeros(user_params['M_max'], dtype=int)
  M_eff_vgmm = np.zeros(user_params['M_max'], dtype=int)
  acc_vgmm_MC = []
  weights_vgmm = np.zeros((user_params['M_max'],nb_tests))
  weights_eff_vgmm = np.zeros((user_params['M_max'],nb_tests))
  err_vgmm = []
  KL_vgmm = []
  # some storage for GMM+AE
  M_gmm_DEC = np.zeros(user_params['M_max'], dtype=int)
  M_eff_gmm_DEC = np.zeros(user_params['M_max'], dtype=int)
  acc_gmm_DEC_MC = []
  weights_gmm_DEC = np.zeros((user_params['M_max'],nb_tests))
  weights_eff_gmm_DEC = np.zeros((user_params['M_max'],nb_tests))
  err_gmm_DEC = []
  KL_gmm_DEC = []
  # some storage for V-GMM+AE
  M_vgmm_DEC = np.zeros(user_params['M_max'], dtype=int)
  M_eff_vgmm_DEC = np.zeros(user_params['M_max'], dtype=int)
  acc_vgmm_DEC_MC = []
  weights_vgmm_DEC = np.zeros((user_params['M_max'],nb_tests))
  weights_eff_vgmm_DEC = np.zeros((user_params['M_max'],nb_tests))
  err_vgmm_DEC = []
  KL_vgmm_DEC = []
  # some storage for GMM+DNN
  M_gmm_DNN = np.zeros(user_params['M_max'], dtype=int)
  M_eff_gmm_DNN = np.zeros(user_params['M_max'], dtype=int)
  acc_gmm_DNN_MC = []
  weights_gmm_DNN = np.zeros((user_params['M_max'],nb_tests))
  weights_eff_gmm_DNN = np.zeros((user_params['M_max'],nb_tests))
  err_gmm_DNN = []
  KL_gmm_DNN = []
  # some storage for V-GMM+DNN
  M_vgmm_DNN = np.zeros(user_params['M_max'], dtype=int)
  M_eff_vgmm_DNN = np.zeros(user_params['M_max'], dtype=int)
  acc_vgmm_DNN_MC = []
  weights_vgmm_DNN = np.zeros((user_params['M_max'],nb_tests))
  weights_eff_vgmm_DNN = np.zeros((user_params['M_max'],nb_tests))
  err_vgmm_DNN = []
  KL_vgmm_DNN = []

  # MC experiment
  IndexError_counter = 0 
  for test_num in range(nb_tests):

    print("MC simulation %d / %d " %(test_num,nb_tests-1) )

    try:
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
      # print("X_train shape: {} \nX_test shape: {} \ny_train shape: {} \ny_test shape: {} \n".format(\
      #       np.shape(X_train),np.shape(X_test),np.shape(y_train),np.shape(y_test)))
      # AE training
      history = manager.ae.AEtrain(X_train, X_test, manager.user_params['epochs'], manager.user_params['batch_size'])  
      # manager.ae.plot_history(history) # plot metrics
      # DNN training 
      if manager.user_params['use_DNN']:
        history = manager.dnn.AEtrain(X_train, X_test, manager.user_params['epochs'], manager.user_params['batch_size'])  
        # manager.dnn.plot_history(history) # plot metrics

      # dictionary of boolean flags related to clustering plots
      want_to_plot_dict = {
        'plot_BIC': False,
        'plot_prediction': False,
        'plot_cm': False
      }  
      # CLUSTERING ON RAW MEASUREMENTS
      # GMM  
      acc_gmm,labels_gmm = manager.clustering(h,labels,'GMM', want_to_plot_dict)
      acc_gmm_MC.append(acc_gmm)
      M_gmm[np.array(labels_gmm).max()] += 1
      weights_gmm[:len(manager.gmm_models_list[0].gmm_model.weights_),test_num] = manager.gmm_models_list[0].gmm_model.weights_
      # V-GMM 
      acc_vgmm,labels_vgmm = manager.clustering(h,labels,'V-GMM',want_to_plot_dict)
      acc_vgmm_MC.append(acc_vgmm)
      M_vgmm[np.array(labels_vgmm).max()] += 1
      weights_vgmm[:len(manager.gmm_models_list[1].gmm_model.weights_),test_num] = manager.gmm_models_list[1].gmm_model.weights_

      # CLUSTERING ON EMBEDDED FEATURES
      # generate deep embedded features on test data
      encoded, decoded,reconstructed_data = manager.ae.AEpredict(X_test)
      # GMM 
      acc_gmm_DEC,labels_gmm_DEC = manager.clustering(encoded, y_test,'GMM',want_to_plot_dict)
      acc_gmm_DEC_MC.append(acc_gmm_DEC)
      M_gmm_DEC[np.array(labels_gmm_DEC).max()] += 1
      weights_gmm_DEC[:len(manager.gmm_models_list[2].gmm_model.weights_),test_num] = manager.gmm_models_list[2].gmm_model.weights_
      # V-GMM 
      acc_vgmm_DEC,labels_vgmm_DEC = manager.clustering(encoded, y_test,'V-GMM',want_to_plot_dict)
      acc_vgmm_DEC_MC.append(acc_vgmm_DEC)
      M_vgmm_DEC[np.array(labels_vgmm_DEC).max()] += 1
      weights_vgmm_DEC[:len(manager.gmm_models_list[3].gmm_model.weights_),test_num] = manager.gmm_models_list[3].gmm_model.weights_

      # CLUSTERING ON TRANSFORMED FEATURES
      if manager.user_params['use_DNN']:
        # generate deep transformed features on test data
        encoded_dnn, decoded_dnn,reconstructed_data_dnn = manager.dnn.AEpredict(X_test)
        # GMM 
        acc_gmm_DNN,labels_gmm_DNN = manager.clustering(encoded_dnn, y_test,'GMM',want_to_plot_dict)
        acc_gmm_DNN_MC.append(acc_gmm_DNN)
        M_gmm_DNN[np.array(labels_gmm_DNN).max()] += 1
        weights_gmm_DNN[:len(manager.gmm_models_list[4].gmm_model.weights_),test_num] = manager.gmm_models_list[4].gmm_model.weights_
        # V-GMM 
        acc_vgmm_DNN,labels_vgmm_DNN = manager.clustering(encoded_dnn, y_test,'V-GMM',want_to_plot_dict)
        acc_vgmm_DNN_MC.append(acc_vgmm_DNN)
        M_vgmm_DNN[np.array(labels_vgmm_DNN).max()] += 1
        weights_vgmm_DNN[:len(manager.gmm_models_list[5].gmm_model.weights_),test_num] = manager.gmm_models_list[5].gmm_model.weights_

      # REPROJECTION ONTO THE UNIT HYPERCUBE 
      trueVertices_counter_list, err_list, vertices_weights_dict_list, KL_list = manager.reprojection()
      for m in range(len(manager.gmm_models_list)):
        if m==0:
          err_gmm.append(err_list[m])
          KL_gmm.append(KL_list[m])
          M_eff_gmm[len(list(vertices_weights_dict_list[m].values()))-1] += 1
          weights_eff_gmm[:len(list(vertices_weights_dict_list[m].values())),test_num] = list(vertices_weights_dict_list[m].values())
        elif m==1:
          err_vgmm.append(err_list[m])
          KL_vgmm.append(KL_list[m])
          M_eff_vgmm[len(list(vertices_weights_dict_list[m].values()))-1] += 1
          weights_eff_vgmm[:len(list(vertices_weights_dict_list[m].values())),test_num] = list(vertices_weights_dict_list[m].values())
        elif m==2:
          err_gmm_DEC.append(err_list[m])
          KL_gmm_DEC.append(KL_list[m])
          M_eff_gmm_DEC[len(list(vertices_weights_dict_list[m].values()))-1] += 1
          weights_eff_gmm_DEC[:len(list(vertices_weights_dict_list[m].values())),test_num] = list(vertices_weights_dict_list[m].values())
        elif m==3:
          err_vgmm_DEC.append(err_list[m])
          KL_vgmm_DEC.append(KL_list[m])
          M_eff_vgmm_DEC[len(list(vertices_weights_dict_list[m].values()))-1] += 1
          weights_eff_vgmm_DEC[:len(list(vertices_weights_dict_list[m].values())),test_num] = list(vertices_weights_dict_list[m].values())
        elif m==4:
          err_gmm_DNN.append(err_list[m])
          KL_gmm_DNN.append(KL_list[m])
          M_eff_gmm_DNN[len(list(vertices_weights_dict_list[m].values()))-1] += 1
          weights_eff_gmm_DNN[:len(list(vertices_weights_dict_list[m].values())),test_num] = list(vertices_weights_dict_list[m].values())
        elif m==5:
          err_vgmm_DNN.append(err_list[m])
          KL_vgmm_DNN.append(KL_list[m])
          M_eff_vgmm_DNN[len(list(vertices_weights_dict_list[m].values()))-1] += 1
          weights_eff_vgmm_DNN[:len(list(vertices_weights_dict_list[m].values())),test_num] = list(vertices_weights_dict_list[m].values())

    except IndexError:
      IndexError_counter += 1

              # ***** RESULTS SAVING *****

  # CLUSTERING PERFORMANCE
  M_est_list = [M_gmm,M_vgmm,M_gmm_DEC,M_vgmm_DEC,M_gmm_DNN,M_vgmm_DNN] \
    if manager.user_params['use_DNN'] else [M_gmm,M_vgmm,M_gmm_DEC,M_vgmm_DEC]
  weights_list = [weights_gmm,weights_vgmm,weights_gmm_DEC,weights_vgmm_DEC,weights_gmm_DNN,weights_vgmm_DNN] \
    if manager.user_params['use_DNN'] else [weights_gmm,weights_vgmm,weights_gmm_DEC,weights_vgmm_DEC]
  acc_list = [acc_gmm_MC,acc_vgmm_MC,acc_gmm_DEC_MC,acc_vgmm_DEC_MC,acc_gmm_DNN_MC,acc_vgmm_DNN_MC] \
    if manager.user_params['use_DNN'] else [acc_gmm_MC,acc_vgmm_MC,acc_gmm_DEC_MC,acc_vgmm_DEC_MC]
  # REPROJECTION ONTO THE UNIT HYPERCUBE 
  M_eff_list = [M_eff_gmm,M_eff_vgmm,M_eff_gmm_DEC,M_eff_vgmm_DEC,M_eff_gmm_DNN,M_eff_vgmm_DNN] \
    if manager.user_params['use_DNN'] else [M_eff_gmm,M_eff_vgmm,M_eff_gmm_DEC,M_eff_vgmm_DEC]
  weights_eff_list =  [weights_eff_gmm,weights_eff_vgmm,weights_eff_gmm_DEC,weights_eff_vgmm_DEC,weights_eff_gmm_DNN,weights_eff_vgmm_DNN] \
    if manager.user_params['use_DNN'] else [weights_eff_gmm,weights_eff_vgmm,weights_eff_gmm_DEC,weights_eff_vgmm_DEC]
  err_list = [err_gmm,err_vgmm,err_gmm_DEC,err_vgmm_DEC,err_gmm_DNN,err_vgmm_DNN] \
    if manager.user_params['use_DNN'] else [err_gmm,err_vgmm,err_gmm_DEC,err_vgmm_DEC]
  KL_list = [KL_gmm,KL_vgmm,KL_gmm_DEC,KL_vgmm_DEC,KL_gmm_DNN,KL_vgmm_DNN] \
    if manager.user_params['use_DNN'] else [KL_gmm,KL_vgmm,KL_gmm_DEC,KL_vgmm_DEC]
  # save results
  results_dict={
    'params':user_params,
    'M_est_list': M_est_list,
    'weights_list': weights_list,
    'acc_list': acc_list,
    'M_eff_list':M_eff_list,
    'weights_eff_list': weights_eff_list,
    'err_list': err_list,
    'KL_list': KL_list,
  }
  user_choice = input('\nDo you want to save the model? \nPress 1 to say YES, any other key otherwise: ') 
  manager.save_MC(user_choice, results_dict)
  print('\nIndexError_counter: ' + str(IndexError_counter))

              # ***** RESULTS PLOTTING *****

# plot results
# manager.plotter.plot_pie(manager.gmm_models_list, results_dict['M_est_list'],r'$\hat{M}$') # estimated number of components
manager.plotter.M_bars_MC(manager.gmm_models_list, results_dict['M_est_list'],r'$\widehat{M}$') # estimated number of components
manager.plotter.weights_bars_MC(manager.gmm_models_list, results_dict['weights_list'],r'$\pi_m$') # weights
manager.plotter.plot_ecdf(manager.gmm_models_list, results_dict['acc_list'],r'$acc$') # accuracy
# manager.plotter.plot_pie(manager.gmm_models_list, results_dict['M_eff_list'],r'$\widehat{M}_{eff}$') # estimated number of effective components
manager.plotter.M_bars_MC(manager.gmm_models_list, results_dict['M_eff_list'],r'$\widehat{M}_{eff}$') # estimated number of components
manager.plotter.weights_bars_MC(manager.gmm_models_list,results_dict['weights_eff_list'],r'$\pi_{m,eff}$') # effective weights
manager.plotter.plot_ecdf(manager.gmm_models_list, results_dict['err_list'],r'$e_r$') # reconstruction error
manager.plotter.plot_ecdf(manager.gmm_models_list, results_dict['KL_list'],r'$D_{KL}$') # KL divergence