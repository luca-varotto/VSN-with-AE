import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

############################################################################################################################

'''
Visualization tool
'''

class Plotter:

    ''' Constructor 
    '''
    def __init__(self, user_params, gmm_models_names, gmm_models_cols):
        self.user_params = user_params
        self.gmm_models_names = gmm_models_names
        self.gmm_models_cols = gmm_models_cols

################### for single test

    '''
    '''
    def weights_bars(self,gmm_models_list,vertices_weights_dict_list=None):
        fig = plt.figure(figsize=(10,6))
        
        # groundtruth pdf
        ax = plt.subplot(2,1,1)
        ax.bar(range(1,self.user_params['M_max']+1),np.hstack((self.user_params['p_events'],\
                np.zeros(self.user_params['M_max']-self.user_params['M']))),align='center',\
                edgecolor=(0.5,0.5,0.5),color='None', width=1.0) # groundtruth
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('2')
        ax.grid(ls = ':', lw = 0.5)
        ax = plt.subplot(2,1,2)
        ax.bar(range(1,self.user_params['M_max']+1),np.hstack((self.user_params['p_events'],\
                np.zeros(self.user_params['M_max']-self.user_params['M']))),align='center',\
                edgecolor=(0.5,0.5,0.5),color='None', width=1.0) # groundtruth
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('2')
        ax.grid(ls = ':', lw = 0.5)
        pdf_max = max(self.user_params['p_events'])
        
        # estimated pdf
        for i in range(len(gmm_models_list)):             
            width = 0.7 if self.gmm_models_names[i][-1]=='E' else (0.5 if self.gmm_models_names[i][-1]=='N' else 0.3)
            if self.gmm_models_names[i][0] != 'V': # GMM models
                ax = plt.subplot(2,1,1)
                if vertices_weights_dict_list is None:
                    ax.bar(range(1,len(gmm_models_list[i].gmm_model.weights_)+1),gmm_models_list[i].gmm_model.weights_,\
                        align='center',label=self.gmm_models_names[i], color=self.gmm_models_cols[i], alpha=0.7, width=width)
                else:
                    ax.bar(range(1,self.user_params['M_max']+1),np.hstack((list(vertices_weights_dict_list[i].values()),\
                        np.zeros(self.user_params['M_max']-len(vertices_weights_dict_list[i].values())))),align='center',\
                            label=self.gmm_models_names[i], color=self.gmm_models_cols[i], alpha=0.7, width=width)
                ax.legend(fontsize=18, loc='upper right',fancybox=True, framealpha=0.5)
                ax.set_xticks(range(1,self.user_params['M_max']+1))
            else: # V-GMM models
                ax = plt.subplot(2,1,2)
                if vertices_weights_dict_list is None:
                    ax.bar(range(1,len(gmm_models_list[i].gmm_model.weights_)+1),gmm_models_list[i].gmm_model.weights_,\
                        align='center',label=self.gmm_models_names[i], color=self.gmm_models_cols[i], alpha=0.7, width=width)
                else:
                    ax.bar(range(1,self.user_params['M_max']+1),np.hstack((list(vertices_weights_dict_list[i].values()),\
                        np.zeros(self.user_params['M_max']-len(vertices_weights_dict_list[i].values())))),align='center', \
                            label=self.gmm_models_names[i], color=self.gmm_models_cols[i], alpha=0.7, width=width)
                ax.legend(fontsize=18, loc='upper right',fancybox=True, framealpha=0.5)
                ax.set_xticks(range(1,self.user_params['M_max']+1))
            if max(gmm_models_list[i].gmm_model.weights_) > pdf_max:
                pdf_max = max(gmm_models_list[i].gmm_model.weights_)
            if vertices_weights_dict_list is None:
                plt.ylabel(r'$\widehat{\pi}_m$',fontsize=18)
            else:
                plt.ylabel(r'$\widehat{\pi}_{m,eff}$',fontsize=18)
            plt.xlabel(r'$m$',fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.ylim([0,pdf_max+0.05])
        plt.tight_layout()
        plt.show()

################### for MC experiment

    ''' pie-chart relative to the number of clusters estimated
    '''
    # def plot_pie(self, gmm_models_list, M_list, suptitle):
    #     fig = plt.figure(figsize=(6,6))
    #     fig.suptitle(suptitle, fontsize=18)
    #     for i in range(len(gmm_models_list)): 
    #         ax = plt.subplot(2,2,i+1)
    #         ax.pie(M_list[i][np.nonzero(M_list[i])[0]], labels=np.nonzero(M_list[i])[0]+1,\
    #             autopct='%1.1f%%', startangle=90,textprops={'fontsize': 14})
    #         plt.title(self.gmm_models_names[i], fontsize=18)
    #         ax.axis('equal')
    #     plt.tight_layout()

    '''
    '''
    def M_bars_MC(self,gmm_models_list, M_list,ylabel):
        fig = plt.figure(figsize=(10,6))
        for i in range(len(gmm_models_list)): 
            width = 0.7 if self.gmm_models_names[i][-1]=='E' else (0.5 if self.gmm_models_names[i][-1]=='N' else 0.3)
            if self.gmm_models_names[i][0] != 'V': # GMM models
                ax = plt.subplot(2,1,1) 
                ax.bar(range(1,self.user_params['M_max']+1),\
                    M_list[i],align='center', alpha=0.7,color=self.gmm_models_cols[i],width=width,label=self.gmm_models_names[i])
                plt.vlines(self.user_params['M'],0,max(M_list[i]), linestyles='--', colors='k')
                ax.legend(fontsize=18, loc='upper right',fancybox=True, framealpha=0.5)
                ax.set_xticks(range(1,self.user_params['M_max']+1))
                ax.patch.set_edgecolor('black')  
                ax.patch.set_linewidth('2')
                ax.grid(ls = ':', lw = 0.5)
            else: # V-GMM models
                ax = plt.subplot(2,1,2)
                ax.bar(range(1,self.user_params['M_max']+1),\
                    M_list[i],align='center', alpha=0.7,color=self.gmm_models_cols[i],width=width,label=self.gmm_models_names[i])
                plt.vlines(self.user_params['M'],0,max(M_list[i]), linestyles='--', colors='k')
                ax.legend(fontsize=18, loc='upper right',fancybox=True, framealpha=0.5)
                ax.set_xticks(range(1,self.user_params['M_max']+1))
                ax.patch.set_edgecolor('black')  
                ax.patch.set_linewidth('2')
                ax.grid(ls = ':', lw = 0.5)
            plt.ylabel(ylabel,fontsize=18)
            plt.xlabel(r'$m$',fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.show()

    ''' bar chart of the weights given to each component
    '''
    def weights_bars_MC(self,gmm_models_list,weights_list,ylabel):
        fig = plt.figure(figsize=(10,9))
        pdf_max = 0
        for i in range(len(gmm_models_list)): 
            ax = plt.subplot(len(gmm_models_list),1,i+1)
            avg_weights = np.mean(weights_list[i], axis= 1)
            std_weights = np.std(weights_list[i], axis= 1)
            ax.bar(range(1,self.user_params['M_max']+1),avg_weights, yerr=std_weights,align='center', \
                color=self.gmm_models_cols[i], alpha=0.5, width=0.5,ecolor='black', capsize=10, label=self.gmm_models_names[i])
            ax.bar(range(1,self.user_params['M_max']+1),np.hstack((self.user_params['p_events'],\
                    np.zeros(self.user_params['M_max']-self.user_params['M']))),align='center',\
                    edgecolor=(0.5,0.5,0.5),color='None', width=0.7) # groundtruth
            ax.legend(fontsize=18, loc='upper right',fancybox=True, framealpha=0.5)
            ax.set_xticks(range(1,self.user_params['M_max']+1))
            if max(avg_weights) > pdf_max:
                pdf_max = max(avg_weights)
            plt.ylabel(ylabel,fontsize=18)
            plt.xlabel(r'$m$',fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.ylim([0,pdf_max+0.05])
            ax.set_xticks(range(1,self.user_params['M_max']+1))
            ax.patch.set_edgecolor('black')  
            ax.patch.set_linewidth('2')
            ax.grid(ls = ':', lw = 0.5)
        plt.tight_layout()
        plt.show()

    ''' plot ECDF of a given performance metric 
    '''
    def plot_ecdf(self,gmm_models_list, performance_list,xlabel):
        fig = plt.figure(figsize=(15,9))
        for i in range(len(gmm_models_list)):
            ecdf = ECDF([item for item in performance_list[i] if item >= 0])
            plt.plot(ecdf.x, ecdf.y, label=self.gmm_models_names[i], c=self.gmm_models_cols[i],linewidth=2)
        plt.legend(loc='lower right', fontsize=25,fancybox=True, framealpha=0.5,ncol=2)
        plt.xlabel(xlabel,fontsize=30)
        plt.ylabel('ECDF',fontsize=30)
        plt.yticks(fontsize=25)
        plt.xticks(fontsize=25)
        plt.ylim([0,1.01])
        ax = plt.gca()
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('2')
        ax.grid(ls = ':', lw = 0.5)
        plt.tight_layout()
        plt.show()
 