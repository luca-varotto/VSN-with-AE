import math
import numpy as np
from tqdm import tqdm
from datetime import datetime
import csv
import pickle
import os
from sklearn.externals import joblib
import tensorflow as tf
from keras.models import load_model

############################################################################################################################

from clusterizer import GMM
from DEC import AE
from plotter import Plotter

############################################################################################################################

''' 
Simulation manager
- create synthetic stimulation model
- synthetic dataset generation
- ...
- variables loading and saving
'''

class Manager:

    ''' Constructor
        user_params: user-defined parameters to handle a simulation 
    ''' 
    def __init__(self, user_params=None,\
                gmm_models_names=['GMM','V-GMM','GMM+AE','V-GMM+AE'], \
                gmm_models_cols=['g','b','y','r']):
        if user_params:
            self.user_params = user_params
            self.ae = AE(self.user_params['N'],\
                    structure_array=self.user_params['structure_array'],\
                    loaded_models=None)  # create the autoencoder model (autoencoder, encoder, decoder) 
            if self.user_params['use_DNN']:
                self.dnn = AE(self.user_params['N'],\
                            structure_array=[self.user_params['N'],self.user_params['N']] ,\
                            loaded_models=None) # create a DNN ...
                self.gmm_models_names = ['GMM','V-GMM','GMM+AE','V-GMM+AE','GMM+DNN','V-GMM+DNN']
                self.gmm_models_cols = ['g','b','y','r','brown','silver']
            else:
                self.gmm_models_names = gmm_models_names
                self.gmm_models_cols = gmm_models_cols
            self.plotter = Plotter(user_params, self.gmm_models_names, self.gmm_models_cols)
            self.gmm_models_list = []  # store all the GMM models used 

    ''' Creation of the stimulation matrix representing the stimulation model
    '''
    def create_T(self):
        T = np.random.randint(0,2,size=(self.user_params['M'],self.user_params['N'])) # stimulation matrix
        # suppose stimulation model (tau) to be injective --> T full row rank
        # check row rank of T; if not full, generate a new one
        while np.linalg.matrix_rank(T) < self.user_params['M']: # suppose M < N
            T = np.random.randint(0,2,size=(self.user_params['M'],self.user_params['N']))
        return T
    
    ''' Generate datatset; each datapoint is a N-dimensional binary vector (N = number of nodes) 
    '''
    # def data_generator(self):
    #     h = np.zeros((self.user_params['n'],self.user_params['N'])) # i-th row = i-th measurement (N-dim binary vector)
    #     labels = [] # groundtruth labels (which is the triggering event of the i-th observation?)
    #     print("*** DATASET GENERATION ***")
    #     for i in tqdm(range(self.user_params['n'])): # i-th observation
    #         # randomly choose one of the M events according to the a-priori distribution
    #         event = np.random.choice(self.user_params['events'], 1, p=self.user_params['p_events'])
    #         labels.append(event)
    #         v_i = np.zeros(self.user_params['N']) # stimulation error for each camera
    #         while np.count_nonzero(h[i,:] == np.zeros(self.user_params['N'])) == self.user_params['N']: # avoid all-zeros vector
    #             for node in range(self.user_params['N']):
    #                 if not np.random.binomial(1,self.user_params['p_succ']): # stimulation error modeled as a Bernoulli process of success probability p_succ
    #                     v_i[node] = 1 if self.user_params['T'][event,node]==0 else -1 # mis-classification or missed detection
    #                 # i-th measurement for all cameras 
    #                 h[i,node] = min(max((self.user_params['T'][event,node]+ v_i[node]),0),1)*\
    #                         np.random.uniform(self.user_params['confidence_range'][0],self.user_params['confidence_range'][1])
    #             if np.count_nonzero(h[i,:] == np.zeros(self.user_params['N'])) == self.user_params['N']:
    #                 h[i,np.random.randint(0,self.user_params['N'])] = 1
    #     return h, labels

    def data_generator(self):
        print("*** DATASET GENERATION ***")
        labels = []
        h = np.zeros((self.user_params['n'],self.user_params['N']))
        p_detect = self.user_params['p_succ']
        p_class = 0.99
        for i in tqdm(range(self.user_params['n'])): # i-th observation
            # randomly choose one of the M events according to the a-priori distribution
            event = np.random.choice(self.user_params['events'], 1, p=self.user_params['p_events'])
            labels.append(event)
            header_candidates = np.arange(0,self.user_params['N'],dtype=int)[np.where(self.user_params['T'][event,:][0]==1)]
            header = np.random.choice(header_candidates,1)
            if np.random.binomial(1,p_class): # stimulation error modeled as a Bernoulli process of success probability p_succ
                event_est = event
            else:
                event_wrong = np.random.randint(low=0,high=self.user_params['M_max'])
                while event_wrong == event:
                    event_wrong = np.random.randint(low=0,high=self.user_params['M_max'])
                event_est = event_wrong
            # event_est = event
            h[i,header] = np.random.uniform(self.user_params['confidence_range'][0],self.user_params['confidence_range'][1])
            for node in range(self.user_params['N']):
                if node != header:
                    frames_num_node = 3
                    frames_counter = 1
                    while frames_counter <= frames_num_node and h[i,node]==0:
                        
                        if frames_counter == frames_num_node and event_est < self.user_params['M'] and self.user_params['T'][event_est,node] == 1:
                            event_frame = event_est
                        else:
                            event_frame = np.random.choice(self.user_params['events'], 1, p=self.user_params['p_events']) #if frames_counter < frames_num_node else event_est
                        # if event_est == event_frame and event_est < self.user_params['M'] and self.user_params['T'][event_est,node] == 1:
                        #     if np.random.binomial(1,self.user_params['p_succ']):
                        #         h[i,node] = np.random.uniform(self.user_params['confidence_range'][0],self.user_params['confidence_range'][1])

                        if event_frame < self.user_params['M'] and self.user_params['T'][event_frame,node] == 1:
                            if event_frame == event_est:
                                if np.random.binomial(1,p_detect) and np.random.binomial(1,p_class): 
                                    h[i,node] = np.random.uniform(self.user_params['confidence_range'][0],self.user_params['confidence_range'][1])
                                    # else:
                                    #     event_wrong = np.random.randint(low=0,high=self.user_params['M'])
                                    #     while event_wrong == event:
                                    #         event_wrong = np.random.randint(low=0,high=self.user_params['M'])
                                    #     h[i,node] = np.random.uniform(self.user_params['confidence_range'][0],self.user_params['confidence_range'][1]) 
                            else:
                                if np.random.binomial(1,p_detect) and np.random.binomial(1,1.0-p_class):
                                        event_wrong = np.random.randint(low=0,high=self.user_params['M_max'])
                                        if event_wrong == event_est:
                                            h[i,node] = np.random.uniform(self.user_params['confidence_range'][0],self.user_params['confidence_range'][1]) 
                        frames_counter += 1

        return h, labels

                            
    '''
    '''
    def clustering(self,h, labels, method_name, want_to_plot_dict):
        M_range = range(1, self.user_params['M_max']+1)
        gmm = GMM() 
        # model fitting 
        if method_name == 'GMM':
            gmm = GMM(gmm.GMM_generation(h, M_range,type='standard', want_to_plot= want_to_plot_dict['plot_BIC'])) # fit on data
        else:
            gmm = GMM(gmm.GMM_generation(h, M_range,type='bayesian', weight_concentration_prior_type='dirichlet_process', \
                weight_concentration_prior=1.0E-7)) # fit on data
        # save the model
        self.gmm_models_list.append(gmm)
        # use the model for clustering
        labels_gmm = gmm.prediction(h,want_to_plot=want_to_plot_dict['plot_prediction']) 
        cm_gmm,acc_gmm,ind_gmm = gmm.confusion_matrix(labels, labels_gmm, want_to_plot_dict['plot_cm']) # confusion matrix
        # sort components according to ind_gmm
        gmm.gmm_model.weights_ = gmm.gmm_model.weights_[ind_gmm[:,1]]
        gmm.gmm_model.means_ = gmm.gmm_model.means_[ind_gmm[:,1]]
        gmm.gmm_model.covariances_ = gmm.gmm_model.covariances_[ind_gmm[:,1]]
        return acc_gmm,labels_gmm

    '''
    '''
    def reprojection(self):
        trueVertices_counter_list = []
        err_list = []
        vertices_weights_dict_list = []
        KL_list = []
        for i in range(len(self.gmm_models_list)):
            # project all GMM centroids onto one of the unit hypercube vertices
            if self.plotter.gmm_models_names[i][-1] == 'M': # GMM or V-GMM
                centroids_projected = np.round(self.gmm_models_list[i].gmm_model.means_).astype(int)
            elif self.plotter.gmm_models_names[i][-1] == 'E': # GMM+AE or V-GMM+AE
                centroids_projected = np.round(self.ae.scaler.inverse_transform( self.ae.decoder.predict(\
                                        self.gmm_models_list[i].gmm_model.means_))).astype(int)
            else:  # GMM+DNN or V-GMM+DNN
                centroids_projected = np.round(self.dnn.scaler.inverse_transform( self.dnn.decoder.predict(\
                                        self.gmm_models_list[i].gmm_model.means_))).astype(int)
            
            trueVertices_counter = np.zeros(self.user_params['M'],dtype=int) # how many clusters are projected onto the true vertices
            err_extra = 0 # sum of the weights of all artifacts (projections on non-existing vertices)
            vertices_weights_dict = {} # dictionary of reconstructed hypercube vertices (keys) and corresponding weights (values)
            for k in range(len(centroids_projected)):
                mu = centroids_projected[k] # k-th centroid of the GMM model
                wrong_reconstruction = True
                for m in range(self.user_params['M']): # compare mu with the m-th row of the stimulation matrix (i.e., m-th vertex hypercube)
                    if np.count_nonzero(self.user_params['T'][m,:] != mu) == 0: # mu equal to the m-th vertex of the unit hypercube
                        trueVertices_counter[m] += 1
                        if str(self.user_params['T'][m,:]) not in vertices_weights_dict.keys(): # m-th vertex not in the dictionary -> create its item
                            vertices_weights_dict[str(self.user_params['T'][m,:])] = self.gmm_models_list[i].gmm_model.weights_[k]
                        else: # m-th vertex already in the dictionary -> update its item
                            vertices_weights_dict[str(self.user_params['T'][m,:])] += self.gmm_models_list[i].gmm_model.weights_[k]
                        wrong_reconstruction = False
                        break
                if wrong_reconstruction: # mu is an artifact
                    err_extra += self.gmm_models_list[i].gmm_model.weights_[k]
                    if str(mu) not in vertices_weights_dict.keys(): # artifact vertex not in the dictionary -> create its item
                        vertices_weights_dict[str(mu)] = self.gmm_models_list[i].gmm_model.weights_[k]
                    else: # artifact vertex already in the dictionary -> update its item
                        vertices_weights_dict[str(mu)] += self.gmm_models_list[i].gmm_model.weights_[k]
            
            err_absence = 0 # sum of weights of all missed active vertices
            absence_counter = 0
            KL = 0 # Kullback-Liebler divergence (only on non-zero probability vertices of both pdfs)
            for m in range(self.user_params['M']):
                if str(self.user_params['T'][m,:]) not in vertices_weights_dict.keys():
                    err_absence +=  self.user_params['p_events'][m]
                    absence_counter +=1
                else:
                    KL += self.user_params['p_events'][m] * \
                        np.log( self.user_params['p_events'][m] /  vertices_weights_dict[str(self.user_params['T'][m,:])])
            trueVertices_counter_list.append(trueVertices_counter)
            # reconstruction error
            err_list.append(err_extra/(self.user_params['M_max']-self.user_params['M']) + err_absence/(self.user_params['M']) )
            vertices_weights_dict_list.append(vertices_weights_dict)
            KL = KL/(self.user_params['M']-absence_counter) if KL>0 else -10
            KL_list.append(KL)
        return trueVertices_counter_list, err_list, vertices_weights_dict_list, KL_list

    ''' Save model, scaler and all parameters
        user_choice: user command to save or not
    '''
    def save(self, user_choice):
        if user_choice=='1': # user command to save
            # define the name of model, scaler and parameters files
            now = datetime.now()
            date_time = now.strftime("%m%d%Y-%H%M%S")
            dir_path = os.path.join('./models/',date_time+"/") 
            os.mkdir(dir_path) 
            autoencoder_filename ='ae.h5' # autoencoder file 
            encoder_filename = 'enc.h5' # encoder file
            decoder_filename ='dec.h5' # decoder file
            scaler_filename = 'scaler.save' # scaler file 
            params_filename_csv ='params.csv' # parameters file (human readable)
            params_filename_pkl = 'params.pkl' # parameters file (easy computer readable)
            
            print("Saving variables...")
            
            self.ae.autoencoder.save(dir_path+autoencoder_filename)
            print("...model saved as "+autoencoder_filename)

            self.ae.encoder.save(dir_path+encoder_filename)
            print("...model saved as "+encoder_filename)

            self.ae.decoder.save(dir_path+decoder_filename)
            print("...model saved as "+decoder_filename)
            
            joblib.dump(self.ae.scaler, dir_path+scaler_filename) 
            print("...scaler saved as "+scaler_filename)
            
            params_file_csv = csv.writer(open(dir_path + params_filename_csv, "w"))
            for key, val in self.user_params.items():
                params_file_csv.writerow([key, val])
            print("...params saved as "+params_filename_csv)

            params_file_pkl = open(dir_path + params_filename_pkl,"wb")
            pickle.dump(self.user_params,params_file_pkl)
            params_file_pkl.close()
            print("...params saved as "+params_filename_pkl)

    ''' Load a pre-trained model, its scaler and all parameters used during its training
        model_folder: name of the folder in which the model is stored
    '''
    def load(self,model_folder):
        dir_path = './models/' + model_folder + "/"

        params_filename = model_folder + '_params.pkl'
        params_file = open(dir_path + params_filename,'rb')
        self.user_params = pickle.load(params_file)
        params_file.close()
        print("\nLoaded params: " + params_filename)
        
        autoencoder_filename = model_folder + '_ae.h5'
        autoencoder = tf.keras.models.load_model(dir_path+autoencoder_filename)
        print("Loaded model: " + autoencoder_filename)

        encoder_filename = model_folder + '_enc.h5'
        encoder = tf.keras.models.load_model(dir_path+encoder_filename)
        print("Loaded model: " + encoder_filename)

        decoder_filename = model_folder + '_dec.h5'
        decoder = tf.keras.models.load_model(dir_path+decoder_filename)
        print("Loaded model: " + decoder_filename)

        loaded_models = {
            'autoencoder':autoencoder,
            'encoder':encoder,
            'decoder':decoder
        }
        self.ae = AE(self.user_params['N'],\
                structure_array=self.user_params['structure_array'],\
                loaded_models=loaded_models)  # create the autoencoder model (autoencoder, encoder, decoder) 
        scaler_filename = model_folder + '_scaler.save'
        self.ae.scaler = joblib.load(dir_path+scaler_filename)
        print("Loaded scaler: " + scaler_filename)

        self.plotter = Plotter(self.user_params, ['GMM','V-GMM','GMM+AE','V-GMM+AE'], ['g','b','y','r'])
        self.gmm_models_list = []

    '''
    '''
    def save_MC(self,user_choice,results_dict):
        if user_choice=='1':
            now = datetime.now()
            date_time = now.strftime("%m%d%Y-%H%M%S")
            dir_path = os.path.join('./results/',date_time+"/") 
            os.mkdir(dir_path)
            for item in results_dict.items():
                name = item[0] + '.pkl'
                file = open(dir_path + name,"wb")
                pickle.dump(item[1],file)
                file.close()
                if item[0]=='params':
                    params_file_csv = csv.writer(open(dir_path + 'params.csv', "w"))
                    for key, val in self.user_params.items():
                        params_file_csv.writerow([key, val])
            
            print("\nRESULTS SAVED")
    
    '''
    '''
    def load_MC(self,results_folder):
        dir_path = './results/' + results_folder + "/"

        params_filename = 'params.pkl'
        params_file = open(dir_path + params_filename,'rb')
        self.user_params = pickle.load(params_file)
        params_file.close()
        print("\nLoaded params: " + params_filename)

        if self.user_params['use_DNN']:
            self.plotter = Plotter(self.user_params, ['GMM','V-GMM','GMM+AE','V-GMM+AE','GMM+DNN','V-GMM+DNN'],\
                 ['g','b','y','r','brown','silver'])
            self.gmm_models_list = [ [] for _ in range(6) ]
        else:
            self.plotter = Plotter(self.user_params, ['GMM','V-GMM','GMM+AE','V-GMM+AE'], ['g','b','y','r'])
            self.gmm_models_list = [ [] for _ in range(4) ]

        results_dict={
            'params':None,
            'M_est_list': None,
            'weights_list': None,
            'M_eff_list':None,
            'acc_list': None,
            'weights_eff_list': None,
            'err_list': None,
            'KL_list': None
        }
        for item in results_dict.items():
            name = item[0] + '.pkl'
            file = open(dir_path + name,'rb')
            results_dict[item[0]] = pickle.load(file)
            params_file.close()
        return results_dict
        