#dependencies
import keras
from keras.models import Model
from keras.layers import Input
from keras.preprocessing import image
from keras.layers import Lambda, Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Layer, Dense
from keras.utils import np_utils
from keras import backend as K
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage.filters import laplace
import tensorflow as tf
from PIL import Image
import re
import os
from sklearn.model_selection import train_test_split
from keras.losses import binary_crossentropy
#------------------

def create_mlae(input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size, loss_function):
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
    mlae.compile(optimizer='adam', loss=loss_function)

    return mlae

def train_mlae(model, X_train, Y_train, X_test, Y_test, epochs, batch_size):
    hist = model.fit(X_train, Y_train,
                        batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(X_test, Y_test))
    return hist

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

#computes bluriness via inverse-variance of Laplacian - higher is blurrier
def blur_factor(arr):
    varLap = laplace(arr).var() #variance of positive Laplacian operator
    return 1/varLap

#----------------------------------------------

#get priors and posts - nothing special here
priors, posts = create_priors_posts('data.csv')


#----------------------------------------------


#FORWARD MLAE MODEL--------------
#given frame t-1, can we predict frame t
print('GATHERING DATA FOR FORWARD MLAE MODEL')

train_indices, test_indices = get_train_test_indices(priors, posts, test_size=0.3)

priors_train, priors_test, posts_train, posts_test = split_training_data(priors, posts, train_indices, test_indices)

#----------------------------------------------

#with MSE
#create and train model
print('INITIALIZING FORWARD MLAE MODEL WITH MSE')
forward_mlae = create_mlae(20000, 1024, 512, 256, 128, 20000, 'mean_squared_error') #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size, loss_function
print('TRAINING FORWARD MLAE MODEL WITH MSE')
hist_forward_mlae = train_mlae(forward_mlae, priors_train, posts_train, priors_test, posts_test, 10, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')

#save loss data
print('SAVING LOSS DATA')
loss_data = np.array(hist_forward_mlae.history['loss'])
np.savetxt('forward_mlae_mse_loss_data.csv', loss_data, delimiter=",")
val_loss_data = np.array(hist_forward_mlae.history['val_loss'])
np.savetxt('forward_mlae_mse_val_loss_data.csv', val_loss_data, delimiter=",")
print('\tDone.')

#plot and save loss figure
print('PLOTTING AND SAVING LOSS FIGURE')
plt.plot(hist_forward_mlae.history['loss'], 'b', label='Training')
plt.plot(hist_forward_mlae.history['val_loss'], 'r', label='Validation')
plt.xlabel('Epoch (Index Starts at 0)')
plt.ylabel('Mean Squared Error')
#plt.legend(['Training', 'Validation'], loc='upper right')
plt.title('Forward MLAE MSE Loss During Training')
plt.savefig('forward_mlae_mse_loss.png')
print('\tDone.')

#save model file
print('SAVING MODEL FILE')
forward_mlae.save('forward_mlae_mse_model.h5')
print('\tDone.')

#save test data
print('SAVING TEST DATA')
np.savetxt('forward_mlae_mse_priors_test.csv', priors_test, delimiter=",")
np.savetxt('forward_mlae_mse_posts_test.csv', posts_test, delimiter=",")
print('\tDone.')

#save predictions
print('SAVING PREDICTIONS')
predictions = forward_mlae.predict(priors_test)
np.savetxt('forward_mlae_mse_predictions.csv', predictions, delimiter=",")
print('\tDone.')

#save blur data
print('SAVING BLUR DATA')
bf_data = pd.DataFrame(np.zeros((len(predictions), 2)), columns = ['bf_predict', 'bf_true'])
posts_test_temp = posts_test.reset_index(drop=True).values
for i in range(len(predictions)):
    bf_predict = blur_factor(predictions[i].reshape((200,100)))
    bf_true = blur_factor(posts_test_temp[i].reshape((200,100)))
    bf_data.loc[i]['bf_predict'] = bf_predict
    bf_data.loc[i]['bf_true'] = bf_true
bf_data.to_csv('forward_mlae_mse_bf_data.csv', index=False)
print('\tDone.')




#----------------------------------------------

#with MSLE
#create and train model
print('INITIALIZING FORWARD MLAE MODEL WITH MSLE')
forward_mlae = create_mlae(20000, 1024, 512, 256, 128, 20000, 'mean_squared_logarithmic_error') #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size, loss_function
print('TRAINING FORWARD MLAE MODEL WITH MSLE')
hist_forward_mlae = train_mlae(forward_mlae, priors_train, posts_train, priors_test, posts_test, 10, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')

#save loss data
print('SAVING LOSS DATA')
loss_data = np.array(hist_forward_mlae.history['loss'])
np.savetxt('forward_mlae_msle_loss_data.csv', loss_data, delimiter=",")
val_loss_data = np.array(hist_forward_mlae.history['val_loss'])
np.savetxt('forward_mlae_msle_val_loss_data.csv', val_loss_data, delimiter=",")
print('\tDone.')

#plot and save loss figure
print('PLOTTING AND SAVING LOSS FIGURE')
plt.plot(hist_forward_mlae.history['loss'], 'b', label='Training')
plt.plot(hist_forward_mlae.history['val_loss'], 'r', label='Validation')
plt.xlabel('Epoch (Index Starts at 0)')
plt.ylabel('Mean Squared Logarithmic Error')
plt.title('Forward MLAE MSLE Loss During Training')
plt.savefig('forward_mlae_msle_loss.png')
print('\tDone.')

#save model file
print('SAVING MODEL FILE')
forward_mlae.save('forward_mlae_msle_model.h5')
print('\tDone.')

#save test data
print('SAVING TEST DATA')
np.savetxt('forward_mlae_msle_priors_test.csv', priors_test, delimiter=",")
np.savetxt('forward_mlae_msle_posts_test.csv', posts_test, delimiter=",")
print('\tDone.')

#save predictions
print('SAVING PREDICTIONS')
predictions = forward_mlae.predict(priors_test)
np.savetxt('forward_mlae_msle_predictions.csv', predictions, delimiter=",")
print('\tDone.')

#save blur data
print('SAVING BLUR DATA')
bf_data = pd.DataFrame(np.zeros((len(predictions), 2)), columns = ['bf_predict', 'bf_true'])
posts_test_temp = posts_test.reset_index(drop=True).values
for i in range(len(predictions)):
    bf_predict = blur_factor(predictions[i].reshape((200,100)))
    bf_true = blur_factor(posts_test_temp[i].reshape((200,100)))
    bf_data.loc[i]['bf_predict'] = bf_predict
    bf_data.loc[i]['bf_true'] = bf_true
bf_data.to_csv('forward_mlae_msle_bf_data.csv', index=False)
print('\tDone.')


#----------------------------------------------

#with SH
#create and train model
print('INITIALIZING FORWARD MLAE MODEL WITH SH')
forward_mlae = create_mlae(20000, 1024, 512, 256, 128, 20000, 'squared_hinge') #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size, loss_function
print('TRAINING FORWARD MLAE MODEL WITH SH')
hist_forward_mlae = train_mlae(forward_mlae, priors_train, posts_train, priors_test, posts_test, 10, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')

#save loss data
print('SAVING LOSS DATA')
loss_data = np.array(hist_forward_mlae.history['loss'])
np.savetxt('forward_mlae_sh_loss_data.csv', loss_data, delimiter=",")
val_loss_data = np.array(hist_forward_mlae.history['val_loss'])
np.savetxt('forward_mlae_sh_val_loss_data.csv', val_loss_data, delimiter=",")
print('\tDone.')

#plot and save loss figure
print('PLOTTING AND SAVING LOSS FIGURE')
plt.plot(hist_forward_mlae.history['loss'], 'b', label='Training')
plt.plot(hist_forward_mlae.history['val_loss'], 'r', label='Validation')
plt.xlabel('Epoch (Index Starts at 0)')
plt.ylabel('Squared Hinge')
plt.title('Forward MLAE SH Loss During Training')
plt.savefig('forward_mlae_sh_loss.png')
print('\tDone.')

#save model file
print('SAVING MODEL FILE')
forward_mlae.save('forward_mlae_sh_model.h5')
print('\tDone.')

#save test data
print('SAVING TEST DATA')
np.savetxt('forward_mlae_sh_priors_test.csv', priors_test, delimiter=",")
np.savetxt('forward_mlae_sh_posts_test.csv', posts_test, delimiter=",")
print('\tDone.')

#save predictions
print('SAVING PREDICTIONS')
predictions = forward_mlae.predict(priors_test)
np.savetxt('forward_mlae_sh_predictions.csv', predictions, delimiter=",")
print('\tDone.')

#save blur data
print('SAVING BLUR DATA')
bf_data = pd.DataFrame(np.zeros((len(predictions), 2)), columns = ['bf_predict', 'bf_true'])
posts_test_temp = posts_test.reset_index(drop=True).values
for i in range(len(predictions)):
    bf_predict = blur_factor(predictions[i].reshape((200,100)))
    bf_true = blur_factor(posts_test_temp[i].reshape((200,100)))
    bf_data.loc[i]['bf_predict'] = bf_predict
    bf_data.loc[i]['bf_true'] = bf_true
bf_data.to_csv('forward_mlae_sh_bf_data.csv', index=False)
print('\tDone.')

#----------------------------------------------

#with CP
#create and train model
print('INITIALIZING FORWARD MLAE MODEL WITH CP')
forward_mlae = create_mlae(20000, 1024, 512, 256, 128, 20000, 'cosine_proximity') #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size, loss_function
print('TRAINING FORWARD MLAE MODEL WITH CP')
hist_forward_mlae = train_mlae(forward_mlae, priors_train, posts_train, priors_test, posts_test, 10, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')

#save loss data
print('SAVING LOSS DATA')
loss_data = np.array(hist_forward_mlae.history['loss'])
np.savetxt('forward_mlae_cp_loss_data.csv', loss_data, delimiter=",")
val_loss_data = np.array(hist_forward_mlae.history['val_loss'])
np.savetxt('forward_mlae_cp_val_loss_data.csv', val_loss_data, delimiter=",")
print('\tDone.')

#plot and save loss figure
print('PLOTTING AND SAVING LOSS FIGURE')
plt.plot(hist_forward_mlae.history['loss'], 'b', label='Training')
plt.plot(hist_forward_mlae.history['val_loss'], 'r', label='Validation')
plt.xlabel('Epoch (Index Starts at 0)')
plt.ylabel('Cosine Proximity')
plt.title('Forward MLAE CP Loss During Training')
plt.savefig('forward_mlae_cp_loss.png')
print('\tDone.')

#save model file
print('SAVING MODEL FILE')
forward_mlae.save('forward_mlae_cp_model.h5')
print('\tDone.')

#save test data
print('SAVING TEST DATA')
np.savetxt('forward_mlae_cp_priors_test.csv', priors_test, delimiter=",")
np.savetxt('forward_mlae_cp_posts_test.csv', posts_test, delimiter=",")
print('\tDone.')

#save predictions
print('SAVING PREDICTIONS')
predictions = forward_mlae.predict(priors_test)
np.savetxt('forward_mlae_cp_predictions.csv', predictions, delimiter=",")
print('\tDone.')

#save blur data
print('SAVING BLUR DATA')
bf_data = pd.DataFrame(np.zeros((len(predictions), 2)), columns = ['bf_predict', 'bf_true'])
posts_test_temp = posts_test.reset_index(drop=True).values
for i in range(len(predictions)):
    bf_predict = blur_factor(predictions[i].reshape((200,100)))
    bf_true = blur_factor(posts_test_temp[i].reshape((200,100)))
    bf_data.loc[i]['bf_predict'] = bf_predict
    bf_data.loc[i]['bf_true'] = bf_true
bf_data.to_csv('forward_mlae_cp_bf_data.csv', index=False)
print('\tDone.')



#BACKWARD MLAE MODEL--------------
#basically: given frame t+1, can we predict frame t
print('\n\nGATHERING DATA FOR BACKWARD MLAE MODEL')


#we want to swap priors and posts for this so we can keep the rest the same as the Forward MLAE model
temp = priors #standard switcheroo
priors = posts
posts = temp
del temp

priors_train, priors_test, posts_train, posts_test = split_training_data(priors, posts, train_indices, test_indices)

#----------------------------------------------

#BACKWARD MLAE MODEL--------------
#basically: given frame t+1, can we predict frame t
print('\n\nGATHERING DATA FOR BACKWARD MLAE MODEL')


#we want to swap priors and posts for this so we can keep the rest the same as the Forward MLAE model
temp = priors #standard switcheroo
priors = posts
posts = temp
del temp

priors_train, priors_test, posts_train, posts_test = split_training_data(priors, posts, train_indices, test_indices)

#----------------------------------------------

#with MSE
#create and train model
print('INITIALIZING BACKWARD MLAE MODEL WITH MSE')
backward_mlae = create_mlae(20000, 1024, 512, 256, 128, 20000, 'mean_squared_error') #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size, loss_function
print('TRAINING BACKWARD MLAE MODEL WITH MSE')
hist_backward_mlae = train_mlae(backward_mlae, priors_train, posts_train, priors_test, posts_test, 10, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')

#save loss data
print('SAVING LOSS DATA')
loss_data = np.array(hist_backward_mlae.history['loss'])
np.savetxt('backward_mlae_mse_loss_data.csv', loss_data, delimiter=",")
val_loss_data = np.array(hist_backward_mlae.history['val_loss'])
np.savetxt('backward_mlae_mse_val_loss_data.csv', val_loss_data, delimiter=",")
print('\tDone.')

#plot and save loss figure
print('PLOTTING AND SAVING LOSS FIGURE')
plt.plot(hist_backward_mlae.history['loss'], 'b', label='Training')
plt.plot(hist_backward_mlae.history['val_loss'], 'r', label='Validation')
plt.xlabel('Epoch (Index Starts at 0)')
plt.ylabel('Mean Squared Error')
#plt.legend(['Training', 'Validation'], loc='upper right')
plt.title('Backward MLAE MSE Loss During Training')
plt.savefig('backward_mlae_mse_loss.png')
print('\tDone.')

#save model file
print('SAVING MODEL FILE')
backward_mlae.save('backward_mlae_mse_model.h5')
print('\tDone.')

#save test data
print('SAVING TEST DATA')
np.savetxt('backward_mlae_mse_priors_test.csv', priors_test, delimiter=",")
np.savetxt('backward_mlae_mse_posts_test.csv', posts_test, delimiter=",")
print('\tDone.')

#save predictions
print('SAVING PREDICTIONS')
predictions = backward_mlae.predict(priors_test)
np.savetxt('backward_mlae_mse_predictions.csv', predictions, delimiter=",")
print('\tDone.')

#save blur data
print('SAVING BLUR DATA')
bf_data = pd.DataFrame(np.zeros((len(predictions), 2)), columns = ['bf_predict', 'bf_true'])
posts_test_temp = posts_test.reset_index(drop=True).values
for i in range(len(predictions)):
    bf_predict = blur_factor(predictions[i].reshape((200,100)))
    bf_true = blur_factor(posts_test_temp[i].reshape((200,100)))
    bf_data.loc[i]['bf_predict'] = bf_predict
    bf_data.loc[i]['bf_true'] = bf_true
bf_data.to_csv('backward_mlae_mse_bf_data.csv', index=False)
print('\tDone.')




#----------------------------------------------

#with MSLE
#create and train model
print('INITIALIZING BACKWARD MLAE MODEL WITH MSLE')
backward_mlae = create_mlae(20000, 1024, 512, 256, 128, 20000, 'mean_squared_logarithmic_error') #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size, loss_function
print('TRAINING BACKWARD MLAE MODEL WITH MSLE')
hist_backward_mlae = train_mlae(backward_mlae, priors_train, posts_train, priors_test, posts_test, 10, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')

#save loss data
print('SAVING LOSS DATA')
loss_data = np.array(hist_backward_mlae.history['loss'])
np.savetxt('backward_mlae_msle_loss_data.csv', loss_data, delimiter=",")
val_loss_data = np.array(hist_backward_mlae.history['val_loss'])
np.savetxt('backward_mlae_msle_val_loss_data.csv', val_loss_data, delimiter=",")
print('\tDone.')

#plot and save loss figure
print('PLOTTING AND SAVING LOSS FIGURE')
plt.plot(hist_backward_mlae.history['loss'], 'b', label='Training')
plt.plot(hist_backward_mlae.history['val_loss'], 'r', label='Validation')
plt.xlabel('Epoch (Index Starts at 0)')
plt.ylabel('Mean Squared Logarithmic Error')
plt.title('Backward MLAE MSLE Loss During Training')
plt.savefig('backward_mlae_msle_loss.png')
print('\tDone.')

#save model file
print('SAVING MODEL FILE')
backward_mlae.save('backward_mlae_msle_model.h5')
print('\tDone.')

#save test data
print('SAVING TEST DATA')
np.savetxt('backward_mlae_msle_priors_test.csv', priors_test, delimiter=",")
np.savetxt('backward_mlae_msle_posts_test.csv', posts_test, delimiter=",")
print('\tDone.')

#save predictions
print('SAVING PREDICTIONS')
predictions = backward_mlae.predict(priors_test)
np.savetxt('backward_mlae_msle_predictions.csv', predictions, delimiter=",")
print('\tDone.')

#save blur data
print('SAVING BLUR DATA')
bf_data = pd.DataFrame(np.zeros((len(predictions), 2)), columns = ['bf_predict', 'bf_true'])
posts_test_temp = posts_test.reset_index(drop=True).values
for i in range(len(predictions)):
    bf_predict = blur_factor(predictions[i].reshape((200,100)))
    bf_true = blur_factor(posts_test_temp[i].reshape((200,100)))
    bf_data.loc[i]['bf_predict'] = bf_predict
    bf_data.loc[i]['bf_true'] = bf_true
bf_data.to_csv('backward_mlae_msle_bf_data.csv', index=False)
print('\tDone.')


#----------------------------------------------

#with SH
#create and train model
print('INITIALIZING BACKWARD MLAE MODEL WITH SH')
backward_mlae = create_mlae(20000, 1024, 512, 256, 128, 20000, 'squared_hinge') #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size, loss_function
print('TRAINING BACKWARD MLAE MODEL WITH SH')
hist_backward_mlae = train_mlae(backward_mlae, priors_train, posts_train, priors_test, posts_test, 10, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')

#save loss data
print('SAVING LOSS DATA')
loss_data = np.array(hist_backward_mlae.history['loss'])
np.savetxt('backward_mlae_sh_loss_data.csv', loss_data, delimiter=",")
val_loss_data = np.array(hist_backward_mlae.history['val_loss'])
np.savetxt('backward_mlae_sh_val_loss_data.csv', val_loss_data, delimiter=",")
print('\tDone.')

#plot and save loss figure
print('PLOTTING AND SAVING LOSS FIGURE')
plt.plot(hist_backward_mlae.history['loss'], 'b', label='Training')
plt.plot(hist_backward_mlae.history['val_loss'], 'r', label='Validation')
plt.xlabel('Epoch (Index Starts at 0)')
plt.ylabel('Squared Hinge')
plt.title('Backward MLAE SH Loss During Training')
plt.savefig('backward_mlae_sh_loss.png')
print('\tDone.')

#save model file
print('SAVING MODEL FILE')
backward_mlae.save('backward_mlae_sh_model.h5')
print('\tDone.')

#save test data
print('SAVING TEST DATA')
np.savetxt('backward_mlae_sh_priors_test.csv', priors_test, delimiter=",")
np.savetxt('backward_mlae_sh_posts_test.csv', posts_test, delimiter=",")
print('\tDone.')

#save predictions
print('SAVING PREDICTIONS')
predictions = backward_mlae.predict(priors_test)
np.savetxt('backward_mlae_sh_predictions.csv', predictions, delimiter=",")
print('\tDone.')

#save blur data
print('SAVING BLUR DATA')
bf_data = pd.DataFrame(np.zeros((len(predictions), 2)), columns = ['bf_predict', 'bf_true'])
posts_test_temp = posts_test.reset_index(drop=True).values
for i in range(len(predictions)):
    bf_predict = blur_factor(predictions[i].reshape((200,100)))
    bf_true = blur_factor(posts_test_temp[i].reshape((200,100)))
    bf_data.loc[i]['bf_predict'] = bf_predict
    bf_data.loc[i]['bf_true'] = bf_true
bf_data.to_csv('backward_mlae_sh_bf_data.csv', index=False)
print('\tDone.')

#----------------------------------------------

#with CP
#create and train model
print('INITIALIZING BACKWARD MLAE MODEL WITH CP')
backward_mlae = create_mlae(20000, 1024, 512, 256, 128, 20000, 'cosine_proximity') #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size, loss_function
print('TRAINING BACKWARD MLAE MODEL WITH CP')
hist_backward_mlae = train_mlae(backward_mlae, priors_train, posts_train, priors_test, posts_test, 10, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')

#save loss data
print('SAVING LOSS DATA')
loss_data = np.array(hist_backward_mlae.history['loss'])
np.savetxt('backward_mlae_cp_loss_data.csv', loss_data, delimiter=",")
val_loss_data = np.array(hist_backward_mlae.history['val_loss'])
np.savetxt('backward_mlae_cp_val_loss_data.csv', val_loss_data, delimiter=",")
print('\tDone.')

#plot and save loss figure
print('PLOTTING AND SAVING LOSS FIGURE')
plt.plot(hist_backward_mlae.history['loss'], 'b', label='Training')
plt.plot(hist_backward_mlae.history['val_loss'], 'r', label='Validation')
plt.xlabel('Epoch (Index Starts at 0)')
plt.ylabel('Cosine Proximity')
plt.title('Backward MLAE CP Loss During Training')
plt.savefig('backward_mlae_cp_loss.png')
print('\tDone.')

#save model file
print('SAVING MODEL FILE')
backward_mlae.save('backward_mlae_cp_model.h5')
print('\tDone.')

#save test data
print('SAVING TEST DATA')
np.savetxt('backward_mlae_cp_priors_test.csv', priors_test, delimiter=",")
np.savetxt('backward_mlae_cp_posts_test.csv', posts_test, delimiter=",")
print('\tDone.')

#save predictions
print('SAVING PREDICTIONS')
predictions = backward_mlae.predict(priors_test)
np.savetxt('backward_mlae_cp_predictions.csv', predictions, delimiter=",")
print('\tDone.')

#save blur data
print('SAVING BLUR DATA')
bf_data = pd.DataFrame(np.zeros((len(predictions), 2)), columns = ['bf_predict', 'bf_true'])
posts_test_temp = posts_test.reset_index(drop=True).values
for i in range(len(predictions)):
    bf_predict = blur_factor(predictions[i].reshape((200,100)))
    bf_true = blur_factor(posts_test_temp[i].reshape((200,100)))
    bf_data.loc[i]['bf_predict'] = bf_predict
    bf_data.loc[i]['bf_true'] = bf_true
bf_data.to_csv('backward_mlae_cp_bf_data.csv', index=False)
print('\tDone.')

#---------------------------------

#BIDIRECTIONAL MLAE MODEL--------------
#basically: given frames t-1, t+1, can we predict frame t
print('\n\nGATHERING DATA FOR BIDIRECTIONAL MLAE MODEL')

# Swap priors/posts back to original
temp = priors #standard switcheroo
priors = posts
posts = temp
del temp

bidirectional_train, bidirectional_response = create_bidirectional_data(priors, posts)
priors_train, priors_test, posts_train, posts_test =  split_training_data(bidirectional_train,
                                                                                      bidirectional_response,
                                                                                      train_indices, test_indices)
#----------------------

#with MSE
#create and train model
print('INITIALIZING BIDIRECTIONAL MLAE MODEL')
bidirectional_mlae = create_mlae(20000*2, 1024, 512, 256, 128, 20000, 'mean_squared_error') #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size, loss_function
print('TRAINING BIDIRECTIONAL MLAE MODEL WITH MSE')
hist_bidirectional_mlae = train_mlae(bidirectional_mlae, priors_train, posts_train, priors_test, posts_test, 10, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')

#save loss data
print('SAVING LOSS DATA')
loss_data = np.array(hist_bidirectional_mlae.history['loss'])
np.savetxt('bidirectional_mlae_mse_loss_data.csv', loss_data, delimiter=",")
val_loss_data = np.array(hist_bidirectional_mlae.history['val_loss'])
np.savetxt('bidirectional_mlae_mse_val_loss_data.csv', val_loss_data, delimiter=",")
print('\tDone.')

#plot and save loss figure
print('PLOTTING AND SAVING LOSS FIGURE')
plt.plot(hist_bidirectional_mlae.history['loss'], 'b', label='Training')
plt.plot(hist_bidirectional_mlae.history['val_loss'], 'r', label='Validation')
plt.xlabel('Epoch (Index Starts at 0)')
plt.ylabel('Mean Squared Error')
#plt.legend(['Training', 'Validation'], loc='upper right')
plt.title('Bidirectional MLAE MSE Loss During Training')
plt.savefig('bidirectional_mlae_mse_loss.png')
print('\tDone.')

#save model file
print('SAVING MODEL FILE')
bidirectional_mlae.save('bidirectional_mlae_mse_model.h5')
print('\tDone.')

#save test data
print('SAVING TEST DATA')
np.savetxt('bidirectional_mlae_mse_priors_test.csv', priors_test, delimiter=",")
np.savetxt('bidirectional_mlae_mse_posts_test.csv', posts_test, delimiter=",")
print('\tDone.')

#save predictions
print('SAVING PREDICTIONS')
predictions = bidirectional_mlae.predict(priors_test)
np.savetxt('bidirectional_mlae_mse_predictions.csv', predictions, delimiter=",")
print('\tDone.')

#save blur data
print('SAVING BLUR DATA')
bf_data = pd.DataFrame(np.zeros((len(predictions), 2)), columns = ['bf_predict', 'bf_true'])
posts_test_temp = posts_test.reset_index(drop=True).values
for i in range(len(predictions)):
    bf_predict = blur_factor(predictions[i].reshape((200,100)))
    bf_true = blur_factor(posts_test_temp[i].reshape((200,100)))
    bf_data.loc[i]['bf_predict'] = bf_predict
    bf_data.loc[i]['bf_true'] = bf_true
bf_data.to_csv('bidirectional_mlae_mse_bf_data.csv', index=False)
print('\tDone.')




#----------------------------------------------

#with MSLE
#create and train model
print('INITIALIZING BIDIRECTIONAL MLAE MODEL WITH MSLE')
bidirectional_mlae = create_mlae(20000*2, 1024, 512, 256, 128, 20000, 'mean_squared_logarithmic_error') #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size, loss_function
print('TRAINING BIDIRECTIONAL MLAE MODEL WITH MSLE')
hist_bidirectional_mlae = train_mlae(bidirectional_mlae, priors_train, posts_train, priors_test, posts_test, 10, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')

#save loss data
print('SAVING LOSS DATA')
loss_data = np.array(hist_bidirectional_mlae.history['loss'])
np.savetxt('bidirectional_mlae_msle_loss_data.csv', loss_data, delimiter=",")
val_loss_data = np.array(hist_bidirectional_mlae.history['val_loss'])
np.savetxt('bidirectional_mlae_msle_val_loss_data.csv', val_loss_data, delimiter=",")
print('\tDone.')

#plot and save loss figure
print('PLOTTING AND SAVING LOSS FIGURE')
plt.plot(hist_bidirectional_mlae.history['loss'], 'b', label='Training')
plt.plot(hist_bidirectional_mlae.history['val_loss'], 'r', label='Validation')
plt.xlabel('Epoch (Index Starts at 0)')
plt.ylabel('Mean Squared Logarithmic Error')
plt.title('Bidirectional MLAE MSLE Loss During Training')
plt.savefig('bidirectional_mlae_msle_loss.png')
print('\tDone.')

#save model file
print('SAVING MODEL FILE')
bidirectional_mlae.save('bidirectional_mlae_msle_model.h5')
print('\tDone.')

#save test data
print('SAVING TEST DATA')
np.savetxt('bidirectional_mlae_msle_priors_test.csv', priors_test, delimiter=",")
np.savetxt('bidirectional_mlae_msle_posts_test.csv', posts_test, delimiter=",")
print('\tDone.')

#save predictions
print('SAVING PREDICTIONS')
predictions = bidirectional_mlae.predict(priors_test)
np.savetxt('bidirectional_mlae_msle_predictions.csv', predictions, delimiter=",")
print('\tDone.')

#save blur data
print('SAVING BLUR DATA')
bf_data = pd.DataFrame(np.zeros((len(predictions), 2)), columns = ['bf_predict', 'bf_true'])
posts_test_temp = posts_test.reset_index(drop=True).values
for i in range(len(predictions)):
    bf_predict = blur_factor(predictions[i].reshape((200,100)))
    bf_true = blur_factor(posts_test_temp[i].reshape((200,100)))
    bf_data.loc[i]['bf_predict'] = bf_predict
    bf_data.loc[i]['bf_true'] = bf_true
bf_data.to_csv('bidirectional_mlae_msle_bf_data.csv', index=False)
print('\tDone.')


#----------------------------------------------

#with SH
#create and train model
print('INITIALIZING BIDIRECTIONAL MLAE MODEL WITH SH')
bidirectional_mlae = create_mlae(20000, 1024, 512, 256, 128, 20000, 'squared_hinge') #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size, loss_function
print('TRAINING BIDIRECTIONAL MLAE MODEL WITH SH')
hist_bidirectional_mlae = train_mlae(bidirectional_mlae, priors_train, posts_train, priors_test, posts_test, 10, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')

#save loss data
print('SAVING LOSS DATA')
loss_data = np.array(hist_bidirectional_mlae.history['loss'])
np.savetxt('bidirectional_mlae_sh_loss_data.csv', loss_data, delimiter=",")
val_loss_data = np.array(hist_bidirectional_mlae.history['val_loss'])
np.savetxt('bidirectional_mlae_sh_val_loss_data.csv', val_loss_data, delimiter=",")
print('\tDone.')

#plot and save loss figure
print('PLOTTING AND SAVING LOSS FIGURE')
plt.plot(hist_bidirectional_mlae.history['loss'], 'b', label='Training')
plt.plot(hist_bidirectional_mlae.history['val_loss'], 'r', label='Validation')
plt.xlabel('Epoch (Index Starts at 0)')
plt.ylabel('Squared Hinge')
plt.title('Bidirectional MLAE SH Loss During Training')
plt.savefig('bidirectional_mlae_sh_loss.png')
print('\tDone.')

#save model file
print('SAVING MODEL FILE')
bidirectional_mlae.save('bidirectional_mlae_sh_model.h5')
print('\tDone.')

#save test data
print('SAVING TEST DATA')
np.savetxt('bidirectional_mlae_sh_priors_test.csv', priors_test, delimiter=",")
np.savetxt('bidirectional_mlae_sh_posts_test.csv', posts_test, delimiter=",")
print('\tDone.')

#save predictions
print('SAVING PREDICTIONS')
predictions = bidirectional_mlae.predict(priors_test)
np.savetxt('bidirectional_mlae_sh_predictions.csv', predictions, delimiter=",")
print('\tDone.')

#save blur data
print('SAVING BLUR DATA')
bf_data = pd.DataFrame(np.zeros((len(predictions), 2)), columns = ['bf_predict', 'bf_true'])
posts_test_temp = posts_test.reset_index(drop=True).values
for i in range(len(predictions)):
    bf_predict = blur_factor(predictions[i].reshape((200,100)))
    bf_true = blur_factor(posts_test_temp[i].reshape((200,100)))
    bf_data.loc[i]['bf_predict'] = bf_predict
    bf_data.loc[i]['bf_true'] = bf_true
bf_data.to_csv('bidirectional_mlae_sh_bf_data.csv', index=False)
print('\tDone.')

#----------------------------------------------

#with CP
#create and train model
print('INITIALIZING BIDIRECTIONAL MLAE MODEL WITH CP')
bidirectional_mlae = create_mlae(20000, 1024, 512, 256, 128, 20000, 'cosine_proximity') #params: input_size, hidden_size1, hidden_size2, hidden_size3, code_size, output_size, loss_function
print('TRAINING BIDIRECTIONAL MLAE MODEL WITH CP')
hist_bidirectional_mlae = train_mlae(bidirectional_mlae, priors_train, posts_train, priors_test, posts_test, 10, 64) #params: model, X_train, Y_train, X_test, Y_test, epochs, batch_size
print('\tDone.')

#save loss data
print('SAVING LOSS DATA')
loss_data = np.array(hist_bidirectional_mlae.history['loss'])
np.savetxt('bidirectional_mlae_cp_loss_data.csv', loss_data, delimiter=",")
val_loss_data = np.array(hist_bidirectional_mlae.history['val_loss'])
np.savetxt('bidirectional_mlae_cp_val_loss_data.csv', val_loss_data, delimiter=",")
print('\tDone.')

#plot and save loss figure
print('PLOTTING AND SAVING LOSS FIGURE')
plt.plot(hist_bidirectional_mlae.history['loss'], 'b', label='Training')
plt.plot(hist_bidirectional_mlae.history['val_loss'], 'r', label='Validation')
plt.xlabel('Epoch (Index Starts at 0)')
plt.ylabel('Cosine Proximity')
plt.title('Bidirectional MLAE CP Loss During Training')
plt.savefig('bidirectional_mlae_cp_loss.png')
print('\tDone.')

#save model file
print('SAVING MODEL FILE')
bidirectional_mlae.save('bidirectional_mlae_cp_model.h5')
print('\tDone.')

#save test data
print('SAVING TEST DATA')
np.savetxt('bidirectional_mlae_cp_priors_test.csv', priors_test, delimiter=",")
np.savetxt('bidirectional_mlae_cp_posts_test.csv', posts_test, delimiter=",")
print('\tDone.')

#save predictions
print('SAVING PREDICTIONS')
predictions = bidirectional_mlae.predict(priors_test)
np.savetxt('bidirectional_mlae_cp_predictions.csv', predictions, delimiter=",")
print('\tDone.')

#save blur data
print('SAVING BLUR DATA')
bf_data = pd.DataFrame(np.zeros((len(predictions), 2)), columns = ['bf_predict', 'bf_true'])
posts_test_temp = posts_test.reset_index(drop=True).values
for i in range(len(predictions)):
    bf_predict = blur_factor(predictions[i].reshape((200,100)))
    bf_true = blur_factor(posts_test_temp[i].reshape((200,100)))
    bf_data.loc[i]['bf_predict'] = bf_predict
    bf_data.loc[i]['bf_true'] = bf_true
bf_data.to_csv('bidirectional_mlae_cp_bf_data.csv', index=False)
print('\tDone.')
