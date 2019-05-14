# DEPENDENCIES
import keras
from keras.models import Model
from keras.preprocessing import image
from keras.layers import Lambda, Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Layer, Dense
from keras.utils import np_utils
import keras.backend as K
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage.filters import laplace
import re
import os
from sklearn.model_selection import train_test_split
from keras.losses import binary_crossentropy
#------------------



# HELPER FUNCTIONS -----------


# DATA PROCESSING FUNCTIONS

def create_priors_posts(file):

    # Reads csv with rows as pixels and separates even/odd rows into prior/post datasets
    
    print('reading image CSV - %s' %(str(file)) )
    df = pd.read_csv(file, header=None)
    df['frame_num'] = pd.to_numeric(df[0].apply(lambda x: re.search(r'\d+', x).group()))
    df = df.sort_values(by='frame_num')
    df = df.drop(['frame_num', 0], axis=1)
    
    print('splitting images into prior/post sets')
    priors = df.iloc[::2].reset_index(drop=True)
    posts = df.iloc[1::2].reset_index(drop=True)

    if len(priors) > len(posts):
        priors = priors[:-1]

    return priors, posts

  
  
def get_train_test_indices(X, y, test_size):    
    
    # Split prior/post dataset based on user-defined split and return frame indices so they can be re-used
    # Only run once!
       
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
    
    train_indices = list(X_train.index)
    test_indices = list(X_test.index)

    return train_indices, test_indices
    


def split_training_data(X, y, train_indices, test_indices):
    
    # Delete train/test index if out of bounds (happens for bidirectional data)

    train_indices = [i for i in train_indices if i <= max(list(X.index))]
    test_indices = [i for i in test_indices if i <= max(list(X.index)) ]    

    # Split dataset based on consistent training/test indices 

    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    

    return X_train, X_test, y_train, y_test



def create_bidirectional_data(priors, posts):
    
    # Shuffle data to match t-1, t+1 frames as training data to label image at time t
    
    bidirectional_train  = priors.copy().shift(-1).reset_index(drop=True)[:-1]
    bidirectional_train = priors.join(bidirectional_train, lsuffix='_t-1', rsuffix='_t+1', how='inner')
    bidirectional_response = posts[:-1]

    return bidirectional_train, bidirectional_response 



def prep_conv_ae_training_data(video_dir, image_dimensions):
    
    # Create priors/posts with numpy dimensions needed for convolutional AE
    # This is a bit hacky but otherwise having difficulty getting shape to be correct
    
    print('Converting images to arrays...')
    image_files = []

    for i in os.listdir(video_dir):
        frame_number = re.search('frame(\d+)', i).group(1)
        frame_image = image.load_img(video_dir + '/' + i, color_mode='grayscale', target_size=image_dimensions)
        frame_array = image.img_to_array(frame_image) / 255
        image_files.append([frame_number, frame_array])
    
    print('Aggregating & sorting image arrays...')    
    df = pd.DataFrame(image_files, columns=['frame_num', 'image'])
    df['frame_num'] = pd.to_numeric(df['frame_num'])

    # need to make sure images are in order
    df = df.sort_values(by='frame_num').reset_index(drop=True)
   
    print('Splitting priors and posts...')
       
    conv_priors = []
    for prior in df.iloc[::2]['image']:
      conv_priors.append(prior)
    
    conv_posts = []
    for post in df.iloc[1::2]['image']:
      conv_posts.append(post)
  
    return np.array(conv_priors), np.array(conv_posts)
     


# AUTOENCODER FUNCTIONS

def create_mlae(input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size):
    

    # Standard NN autoencoder (used for forward, backward, and bi-directional)

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



def create_conv_ae(input_shape, filters):

    # Convolutional autoencoder
    
    input_img = Input(input_shape)  

    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    conv_autoencoder = Model(input_img, decoded)
    conv_autoencoder.compile(optimizer='adadelta', loss='mse')
    return conv_autoencoder


# Variational autoencoder (VAE) sampling helper function

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

  


def create_vae(original_dim, intermediate_dim, latent_dim):

    # Variational autoencoder

    input_shape = (original_dim, )

    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    # VAE custom loss function (variables not found if outside this function)
    def vae_loss(inputs, outputs):
          reconstruction_loss = binary_crossentropy(inputs, outputs)
          reconstruction_loss *= original_dim
          kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
          kl_loss = K.sum(kl_loss, axis=-1)
          kl_loss *= -0.5
          vae_loss = K.mean(reconstruction_loss + kl_loss)
          return vae_loss
    
    models = (encoder, decoder)
    vae.compile(optimizer='adam', loss=vae_loss)
    return models, vae



# Helper function to kick off training for any model type 

def train_mlae(model, X_train, Y_train, X_test, Y_test, epochs, batch_size):
    hist = model.fit(X_train, Y_train,
                        batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(X_test, Y_test))
    return hist





# RUN ACTUAL MODELS

#FORWARD MLAE MODEL--------------
#basically: given frame t-1, can we predict frame t
print('\n\nGATHERING DATA FOR FORWARD MLAE MODEL')

# Use subset for testing 

#priors, posts = create_priors_posts('small_data.csv')

priors, posts = create_priors_posts('data.csv')


train_indices, test_indices = get_train_test_indices(priors, posts, test_size=0.3)

priors_train, priors_test, posts_train, posts_test = split_training_data(priors, posts, train_indices, test_indices)


#create and train model
print('\nINITIALIZING FORWARD MLAE MODEL')
forward_mlae = create_mlae(20000, 1024, 512, 256, 128, 20000) #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size
print('\nTRAINING FORWARD MLAE MODEL')
hist_forward_mlae = train_mlae(forward_mlae, priors_train, posts_train, priors_test, posts_test, 1, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')
#--------------



#BACKWARD MLAE MODEL--------------
#basically: given frame t+1, can we predict frame t
print('\n\nGATHERING DATA FOR BACKWARD MLAE MODEL')


#we want to swap priors and posts for this so we can keep the rest the same as the Forward MLAE model
temp = priors #standard switcheroo
priors = posts
posts = temp
del temp

priors_train, priors_test, posts_train, posts_test = split_training_data(priors, posts, train_indices, test_indices)

#create and train model
print('\nINITIALIZING BACKWARD MLAE MODEL')
backward_mlae = create_mlae(20000, 1024, 512, 256, 128, 20000) #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size
print('\nTRAINING BACKWARD MLAE MODEL')
hist_backward_mlae = train_mlae(backward_mlae, priors_train, posts_train, priors_test, posts_test, 1, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')
#--------------


#BIDIRECTIONAL MLAE MODEL--------------
#basically: given frames t-1, t+1, can we predict frame t
print('\n\nGATHERING DATA FOR BIDIRECTIONAL MLAE MODEL')

# Swap priors/posts back to original
temp = priors #standard switcheroo
priors = posts
posts = temp
del temp

bidirectional_train, bidirectional_response = create_bidirectional_data(priors, posts)
bi_priors_train, bi_priors_test, bi_posts_train, bi_posts_test =  split_training_data(bidirectional_train, 
                                                                                      bidirectional_response, 
                                                                                      train_indices, test_indices)
#create and train model
print('\nINITIALIZING BIDIRECTIONAL MLAE MODEL')
bidirectional_mlae = create_mlae(20000*2, 1024, 512, 256, 128, 20000) #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size
print('\nTRAINING BIDIRECTIONAL MLAE MODEL')
hist_bidirectional_mlae = train_mlae(bidirectional_mlae, bi_priors_train, bi_posts_train, bi_priors_test, bi_posts_test, 1, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')
#--------------


#CONVOLUTIONAL AUTOENCODER--------------

print('\n\nGATHERING DATA FOR CONVOLUTIONAL AE MODEL')

# Run custom conv_ae data pipeline
conv_priors, conv_posts = prep_conv_ae_training_data(video_dir='Frames_bambi', image_dimensions=(200, 100))

conv_priors_train, conv_posts_train = conv_priors[(train_indices)], conv_posts[(train_indices)]
conv_priors_test, conv_posts_test = conv_priors[(test_indices)], conv_posts[(test_indices)]

# Free up some memory
del conv_priors
del conv_posts

#create and train model
print('\nINITIALIZING CONVOLUTIONAL AE MODEL')
conv_autoencoder = create_conv_ae(input_shape=(200, 100, 1), filters=32)
print('\nTRAINING CONVOLUTIONAL AE MODEL')
hist_convolutional_ae = train_mlae(conv_autoencoder, conv_priors_train, conv_posts_train, conv_priors_test, conv_posts_test, epochs=1, batch_size=1)
print('\tDone.')
#--------------



#VARIATIONAL AUTOENCODER--------------
print('\n\nGATHERING DATA FOR VARIATIONAL AE MODEL')
priors_train, priors_test, posts_train, posts_test = split_training_data(priors, posts, train_indices, test_indices)

print('\nINITIALIZING VARIATIONAL AE MODEL')
models, vae = create_vae(original_dim=20000, intermediate_dim=512, latent_dim=3)
print('\nTRAINING VARIATIONAL AE MODEL')
hist_vae = train_mlae(vae, priors_train, posts_train, priors_test, posts_test, epochs=1, batch_size=1)
print('\tDone.')

#--------------



