#dependencies
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Layer, Dense
from keras.utils import np_utils
import keras.backend as K
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage.filters import laplace
#------------------



#functions -----------
def create_mlae(input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size):
    input_layer = Input(shape=(input_size,)) #input
    hidden_1 = Dense(hidden_size1, activation='relu')(input_layer) #encoder
    hidden_2 = Dense(hidden_size2, activation='relu')(hidden_1) #encoder
    hidden_3 = Dense(hidden_size3, activation='relu')(hidden_2) #encoder
    code = Dense(code_size, activation='relu')(hidden_3) #code
    hidden_4 = Dense(hidden_size3, activation='relu')(code) #decoder
    hidden_5 = Dense(hidden_size2, activation='relu')(hidden_4) #decoder
    hidden_6 = Dense(hidden_size1, activation='relu')(hidden_5) #decoder
    output_layer = Dense(output_size, activation='sigmoid')(hidden_6) #output

    mlae = Model(inputs=input_layer, outputs=output_layer)
    mlae.compile(optimizer='adam', loss='mse')

    return mlae

def create_rbmlae(input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size):
    input_layer = Input(shape=(input_size,)) #input
    hidden_1 = Dense(hidden_size1, activation='relu')(input_layer) #encoder
    hidden_2 = Dense(hidden_size2, activation='relu')(hidden_1) #encoder
    hidden_3 = Dense(hidden_size3, activation='relu')(hidden_2) #encoder
    code = Dense(code_size, activation='relu')(hidden_3) #code
    hidden_4 = Dense(hidden_size3, activation='relu')(code) #decoder
    hidden_5 = Dense(hidden_size2, activation='relu')(hidden_4) #decoder
    hidden_6 = Dense(hidden_size1, activation='relu')(hidden_5) #decoder
    output_layer = Dense(output_size, activation='sigmoid')(hidden_6) #output

    mlae = Model(inputs=input_layer, outputs=output_layer)
    mlae.compile(optimizer='adam', loss=blur_adjusted_mse) #'mse'

    return mlae

def train_mlae(model, X_train, Y_train, X_test, Y_test, epochs, batch_size):
    hist = model.fit(X_train, Y_train,
                        batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(X_test, Y_test))
    return hist

def create_priors_posts(file):
    df = pd.read_csv(file)
    df = df[[i for i in df if i!='Mulan - Ho_frame1']].values # think frame 1 is just black, so there ends up being some divide by 0 thing

    priors = [df[i] for i in range(len(df)) if i%2==0][:-1]
    posts = [df[i] for i in range(len(df)) if i%2==1]

    return priors, posts

def blur_adjusted_mse(y_true, y_pred):
    #print(y_true.shape, y_pred.shape)
    arr_pred = K.eval(y_pred).reshape((200,100))
    arr_true = K.eval(y_pred).reshape((200,100))
    loss_blur = abs(blur_factor(arr_pred) - blur_factor(arr_true))
    loss_mse = K.mean((y_true - y_pred)**2)
    return loss_mse + loss_mse

#computes bluriness via inverse-variance of Laplacian - higher is blurrier
def blur_factor(arr, k=117): #experimentation suggests a factor k~=117 brings the average blur factor to 1
    varLap = laplace(arr).var() #variance of positive Laplacian operator
    return k/varLap
#------------


#get priors and posts - nothing special here
priors, posts = create_priors_posts('data.csv')



#FORWARD MLAE MODEL--------------
#basically: given frame t-1, can we predict frame t
print('GATHERING DATA FOR FORWARD MLAE MODEL')

#randomize priors and posts
indices = np.arange(len(priors))
np.random.shuffle(indices)
#this just a hacky way of getting indices for a 70/30 train/test split
train_indices = [i for i in indices if random.random()<0.7]
test_indices = [i for i in indices if i not in train_indices]

priors_train = np.array([priors[i].tolist() for i in train_indices])
posts_train = np.array([posts[i].tolist() for i in train_indices])
priors_test = np.array([priors[i].tolist() for i in test_indices])
posts_test = np.array([posts[i].tolist() for i in test_indices])

#create and train model
print('INITIALIZING FORWARD MLAE MODEL')
forward_mlae = create_mlae(20000, 1024, 512, 256, 128, 20000) #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size
print('TRAINING FORWARD MLAE MODEL')
hist_forward_mlae = train_mlae(forward_mlae, priors_train, posts_train, priors_test, posts_test, 1, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')
#--------------



#BACKWARD MLAE MODEL--------------
#basically: given frame t+1, can we predict frame t
print('GATHERING DATA FOR FORWARD MLAE MODEL')

#we want to swap priors and posts for this so we can keep the rest the same as the Forward MLAE model
temp = priors #standard switcheroo
priors = posts
posts = temp
del temp #don't need this anymore

#the rest should be the same:

priors_train = np.array([priors[i].tolist() for i in train_indices])
posts_train = np.array([posts[i].tolist() for i in train_indices])
priors_test = np.array([priors[i].tolist() for i in test_indices])
posts_test = np.array([posts[i].tolist() for i in test_indices])

#create and train model
print('INITIALIZING BACKWARD MLAE MODEL')
backward_mlae = create_mlae(20000, 1024, 512, 256, 128, 20000) #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size
print('TRAINING BACKWARD MLAE MODEL')
hist_backward_mlae = train_mlae(backward_mlae, priors_train, posts_train, priors_test, posts_test, 1, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')
#--------------



#BIDIRECTIONAL MLAE MODEL--------------
#basically: given frames t-1, t+1, can we predict frame t
print('GATHERING DATA FOR BIDIRECTIONAL MLAE MODEL')

#swap priors and posts back to their original places
temp = priors #standard switcheroo
priors = posts
posts = temp
del temp #don't need this anymore

#now we want to extend the priors to include both t-1 and t+1
priors = [np.array(priors[i].tolist() + priors[i+1].tolist()) for i in range(len(priors)-1)]
del posts[len(priors)-1]

#the rest should be the same, except for doubling the input size of the MLAE

priors_train = np.array([priors[i].tolist() for i in train_indices])
posts_train = np.array([posts[i].tolist() for i in train_indices])
priors_test = np.array([priors[i].tolist() for i in test_indices])
posts_test = np.array([posts[i].tolist() for i in test_indices])

#create and train model
print('INITIALIZING BIDIRECTIONAL MLAE MODEL')
bidirectional_mlae = create_mlae(20000*2, 1024, 512, 256, 128, 20000) #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size
print('TRAINING BIDIRECTIONAL MLAE MODEL')
hist_bidirectional_mlae = train_mlae(bidirectional_mlae, priors_train, posts_train, priors_test, posts_test, 1, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')
#--------------
