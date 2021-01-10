from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

############################################################################################################################

# AutoEncoder object

class AE:

    ''' Constructor
    '''
    def __init__(self,input_dim,structure_array=None, loaded_model=None):
        self.input_dim = input_dim # number of input nodes
        self.structure_array = structure_array # len(structure_array) = # layers; i-th entry = # nodes i-th layer 
        self.encoder, self.decoder, self.autoencoder = self.AEstructure(loaded_model)
        
    ''' Define the structure of the AutoEncoder
    '''
    def AEstructure(self, loaded_model=None):
        
        if loaded_model is None:
            # input placeholder
            inpt = Input(shape=(self.input_dim,))

            # encoded representation of the input
            encoded = Dense(self.structure_array[0], activation='relu')(inpt)        
            for i in range(1,len(self.structure_array)):
                encoded = Dense(self.structure_array[i], activation='relu')(encoded)

            # this model maps an input to its encoded representation
            encoder = Model(inpt, encoded)

            # lossy reconstruction of the input
            decoded = Dense(self.structure_array[-2], activation='relu')(encoded)
            for i in range(len(self.structure_array)-2):
                decoded = Dense(self.structure_array[-(i+3)], activation='relu')(decoded)
            decoded = Dense(self.input_dim, activation='sigmoid')(decoded)

            # this model maps an input to its reconstruction
            autoencoder = Model(inpt, decoded)
            autoencoder.compile(optimizer='adadelta', loss='mse')

            # create a placeholder for an encoded input
            decoder_input = Input(shape=(self.structure_array[-1],))
            # retrieve the last layer of the autoencoder model
            # decoded_output = autoencoder.layers[-1]
            decoder_output = autoencoder.layers[-len(self.structure_array)](decoder_input)
            for i in range(1,len(self.structure_array)):
                decoder_output = autoencoder.layers[-len(self.structure_array)+i](decoder_output)
            # create the decoder model
            decoder = Model(decoder_input, decoder_output)      
        else:
            autoencoder = loaded_model

            num_layers = len(autoencoder.layers)
            encoder = Model(autoencoder.layers[0].input, autoencoder.layers[int((num_layers-1)/2)].output)
            
            decoder_input = Input(shape=(encoder.output_shape[-1],) )
            decoder_output = autoencoder.layers[-int((num_layers-1)/2)](decoder_input)
            for i in range(1,int((num_layers-1)/2)):
                decoder_output = autoencoder.layers[-int((num_layers-1)/2)+i](decoder_output)
            decoder = Model(decoder_input, decoder_output)
            # decoder = None

        # autoencoder.summary()
        # encoder.summary()
        # decoder.summary()

        return encoder, decoder, autoencoder

    ''' Train/test data splitting and normalize
    X: datapoints
    y: labels
    '''
    def data_preproc(self, X, y):
        
        # 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
        
        # define scaler
        scaler = MinMaxScaler(feature_range = (0, 1))
        X_train = scaler.fit_transform(X_train) 
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, scaler

    ''' Train the autoencoder
    '''
    def AEtrain(self,X_train, X_val ,epochs=100,batch_size=256):
        history = self.autoencoder.fit(X_train, X_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_val, X_val))
        return history

    ''' Plot history
    '''
    def plot_history(self,history):
        plt.figure()
        plt.plot(history.history['loss'],label='training loss')
        plt.plot(history.history['val_loss'],label='validation loss')
        plt.grid()
        plt.title('Training vs validation loss')
        plt.legend()
        plt.show()

    ''' Save model
    '''
    def AEsave(self,name):
        self.autoencoder.save(name)

    ''' Predict 
    '''
    def AEpredict(self, x_test):
        encoded_data = self.encoder.predict(x_test) 
        decoded_data = self.decoder.predict(encoded_data) if self.decoder else None
        reconstructed_data = self.autoencoder.predict(x_test) 
        return encoded_data, decoded_data,reconstructed_data









