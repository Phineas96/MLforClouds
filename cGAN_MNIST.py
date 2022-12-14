#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:32:32 2022

@author: mrosenberger
"""

import time
import os
import numpy as np
import tensorflow as tf
#import tensorflow.keras as keras
#from tf.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
#from PIL import Image 
#import h5py
#import json
from IPython import display
#import ray
#import datetime

from keras import layers
#from keras import models

from keras.datasets import mnist
from keras.utils import to_categorical

'''
relu als activation auch in Convolutional layers soll Training beschleunigen laut Krizhevsky2012
'''

#os.environ["OMP_NUM_THREADS"] = '15'

tf.config.threading.set_intra_op_parallelism_threads(num_threads = 10)

tf.config.threading.set_inter_op_parallelism_threads(1)

# Funktion die den Generator, i.e. den Artist, erstellt
def make_generator_model():
    model = tf.keras.Sequential()
    # model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    # model.add(layers.Reshape((7, 7, 256)))
    #assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation = 'relu', input_shape = [7,7,256]))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation = 'relu'))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


# Funktion die den Diskriminator, i.e. den Kritiker, erstellt
# Output des Diskriminators ist <0 wenn fake image und >0 wenn real image
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', activation = 'relu',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation = 'relu'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(11, activation = 'sigmoid'))

    return model

def make_preparation_model(final_shape= (7,7)):
    in_noise = tf.keras.Input(shape=(100,), name='Generator-Noise-Input-Layer')
    g = layers.Dense(final_shape[0]*final_shape[1]*248, use_bias=False)(in_noise)
    g = layers.BatchNormalization()(g)
    g = layers.LeakyReLU()(g)
    g = layers.Reshape((final_shape[0], final_shape[1], 248))(g)
    
    
    in_label = tf.keras.Input(shape=(10,), name='Generator-Label-Input-Layer')
    lbl = layers.Dense(final_shape[0]*final_shape[1]*8, use_bias=False)(in_label)
    lbl = layers.Reshape((final_shape[0],final_shape[1],8))(lbl)
    
    output = layers.Concatenate(name = 'Concatenation-Layer')([g,lbl])
    
    model = tf.keras.Model([in_noise, in_label], output, name='Preparator')

    return model

# Loss im Diskriminator
# Testergebnis bei echtem Bild mit 1er Vektor und bei falschem Bild mit 0er Vektor verglichen
# Unterteilung des Outputs in bool (letzter Eintrag) und label (die zehn davor)
def discriminator_loss(real_output, real_labels, fake_output, fake_labels, wgt):
    real_loss = cross_entropy(tf.ones_like(real_output[:,-1]) + 0.05 * tf.random.uniform((real_output[:,-1]).shape), real_output[:,-1])
    #real_loss_label = cross_entropy(real_labels, real_output[:,:-1])
    
    fake_loss = cross_entropy(tf.zeros_like(fake_output[:,-1])+ 0.05 * tf.random.uniform((fake_output[:,-1]).shape), fake_output[:,-1])
    #fake_loss_label = cross_entropy(fake_labels, fake_output[:,:-1])
    
    total_loss = 1.2*(real_loss + fake_loss)# + 0.8*(real_loss_label + fake_loss_label*wgt)
    
    return total_loss

# Loss des Generators
# will den Kritiker austricksen, Testergebnis des Diskriminators für fake Bilder wird also mit 1er Vektor verglichen
def generator_loss(fake_output, fake_labels, wgt):
    loss_bool = cross_entropy(tf.ones_like(fake_output[:,-1]), fake_output[:,-1])
    #loss_label = cross_entropy(fake_labels, fake_output[:,:-1])
        
    return loss_bool# + loss_label*wgt

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, real_labels, noise, fake_labels, wgt):
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
    
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
    
        gen_loss = generator_loss(fake_output, fake_labels, wgt)
        disc_loss = discriminator_loss(real_output, real_labels, fake_output, fake_labels, wgt)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    train_real_acc.update_state(y_true = real_labels, y_pred = real_output[:,:-1])
    train_fake_acc.update_state(y_true = fake_labels, y_pred = fake_output[:,:-1])
    
    return gen_loss, disc_loss

#@tf.function
def train_step_labels(images, real_labels, noise, fake_labels, epoch, batch_counter):
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        real_labels_pred = real_output[:,:-1]
        fake_labels_pred = fake_output[:,:-1]
            
        # real_classes_true = np.argmax(real_labels_true, axis = 1)
        # real_classes_pred = np.argmax(real_labels_pred, axis = 1)
        
        real_rat_correct = (np.argmax(real_labels, axis = 1) == np.argmax(real_labels_pred, axis = 1)).sum()/BATCH_SIZE
        real_pmax = np.max(real_labels_pred)

        fake_rat_correct = (np.argmax(fake_labels, axis = 1) == np.argmax(fake_labels_pred, axis = 1)).sum()/BATCH_SIZE
        fake_pmax = np.max(fake_labels_pred)

        log_file_misc.write(str(epoch) + ',' + str(batch_counter) + ',' + 
                            str(real_rat_correct) + ',' + str(real_pmax) + ',' + 
                            str(fake_rat_correct) + ',' + str(fake_pmax) + '\n')
    
        # Real Input data
        if  real_rat_correct > thresh_quant and  real_pmax > thresh_prob:
            real_loss = cross_entropy(y_true = real_labels, y_pred = real_labels_pred)
        else:
            real_loss = 2*cross_entropy(y_true = real_labels, y_pred = real_labels_pred)
        
        # Fake Input data
        if fake_rat_correct > thresh_quant and fake_pmax > thresh_prob:
            fake_loss = cross_entropy(y_true = fake_labels, y_pred = fake_labels_pred)
        else:
            fake_loss = 2*cross_entropy(y_true = fake_labels, y_pred = fake_labels_pred)
            
    
        disc_loss = real_loss + fake_loss
        gen_loss = fake_loss
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


    return gen_loss, disc_loss


def train(dataset, labels, epochs):
    
    for epoch in range(epochs):
        #epoch+=10
        start = time.time()
    
        batch_counter = 1
        for image_batch, label_batch in zip(dataset, labels):
            noise = tf.random.normal([BATCH_SIZE, noise_dim])
            fake_labels = to_categorical(np.random.randint(0, 10, size = BATCH_SIZE), num_classes=10) + 0.05*tf.random.uniform((BATCH_SIZE,10))
            
            real_labels = to_categorical(label_batch, num_classes=10)+ 0.05*tf.random.uniform((np.shape(label_batch)[0],10))
            
            # ersetzt die ersten Generator Schichten
            noise = prep([noise, fake_labels])
            
            #wgt = 2 - 0.8*epoch/(epochs-1)
            wgt = 1
            gen_loss, disc_loss = train_step(image_batch, real_labels, noise, fake_labels, wgt)
            
            gen_loss_labels, disc_loss_labels = train_step_labels(image_batch, real_labels, noise, fake_labels, epoch, batch_counter)
            
            log_file_train.write(str(epoch) + ',' + str(batch_counter) + ',' + 
                                 str(gen_loss.numpy()) + ',' + str(gen_loss_labels.numpy()) + ',' + 
                                 str(disc_loss.numpy()) + ',' + str(disc_loss_labels.numpy()) + 
                                 '\n')
            
            batch_counter += 1
        
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                  epoch + 1,
                                  seed)
        
        log_file_trainacc.write(str(epoch) + ',' + str(float(train_real_acc.result())) + ',' + str(float(train_fake_acc.result())) + '\n')
        
        train_real_acc.reset_states()
        train_fake_acc.reset_states()
        
        validation(epoch+1)
        
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            # generator.save(os.path.join(checkpoint_dir, 'generator_{:04d}epochs'.format(epoch+1)))
            # discriminator.save(os.path.join(checkpoint_dir, 'discriminator_{:04d}epochs'.format(epoch+1)))
        
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print('G loss bool = ', gen_loss.numpy())
        print('D loss bool = ', disc_loss.numpy())
        
    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                              epochs,
                              seed)
    
    
    # model.save speichert das fertige Modell ab, die Resultate sind ident.
    # checkpoint.save specihert checkpoints, wie das mit dem restore ist muss ich mir erst ansehen
    
    # save final model
    checkpoint.save(file_prefix = checkpoint_prefix)
    
    # save it again
    generator.save(os.path.join(checkpoint_dir, 'generator_{:04d}epochs'.format(epochs)))
    discriminator.save(os.path.join(checkpoint_dir, 'discriminator_{:04d}epochs'.format(epochs)))

    
def validation(epoch):
    
    val_ints = np.random.randint(0, np.shape(test_images)[0], size = BATCH_SIZE_VAL)
    val_img_batch = test_images[val_ints]
    val_label_batch_real = to_categorical(test_labels[val_ints], num_classes=10)
    
    val_pred = discriminator(val_img_batch, training = False)
    
    val_label_batch_pred = np.array(val_pred[:,:-1])
    val_bool_batch_pred = np.array(val_pred[:,-1])
    
    val_label_loss = cross_entropy(y_true = val_label_batch_real, y_pred = val_label_batch_pred).numpy()
    val_bool_loss = cross_entropy(y_true = tf.ones_like(val_bool_batch_pred), y_pred = val_bool_batch_pred).numpy()
    
    #val_label_acc = (np.argmax(val_label_batch_real, axis = 1) == np.argmax(val_label_batch_pred, axis = 1)).sum()/BATCH_SIZE_VAL
    val_acc.update_state(y_true = val_label_batch_real, y_pred = val_label_batch_pred)
    val_label_acc = float(val_acc.result())
    val_acc.reset_states()
    val_bool_acc = (val_bool_batch_pred > 0.75).sum()/BATCH_SIZE_VAL
    
    log_file_val.write(str(epoch) + ',' + str(val_label_loss) + ',' + str(val_label_acc) + ',' + 
                       str(val_bool_loss) + ',' + str(val_bool_acc) + '\n')
    
    
def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
        plt.title(ints[i])

    plt.savefig(os.path.join(checkpoint_dir, 'image_at_epoch_{:04d}.png'.format(epoch)), bbox_inches = 'tight')
    plt.show()
  

# def display_image(epoch_no):
#     return Image.open(os.path.join(checkpoint_dir, 'image_at_epoch_{:04d}.png'.format(epoch_no)))


# Initiieren von Generator & Diskriminator
generator = make_generator_model()
discriminator = make_discriminator_model()
prep = make_preparation_model()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE) #from_logits=True

# Die Optimizer Funktion für beide Netzwerke
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # hier kann auch SGD versucht werden

train_real_acc = tf.keras.metrics.CategoricalAccuracy()
train_fake_acc = tf.keras.metrics.CategoricalAccuracy()
val_acc = tf.keras.metrics.CategoricalAccuracy()

home_dir = '/mnt/users/staff/mrosenberger' # am jet01: '/jetfs/home/mrosenberger'
checkpoint_dir = os.path.join(home_dir, 'CNNPictures/MNIST/cGAN5')
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# Details des Trainingsprozesses
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

thresh_prob = tf.constant(0.7) 
thresh_quant = tf.constant(0.2)


# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

ints = np.random.randint(0, 10, size = num_examples_to_generate)
seed_labels = to_categorical(ints, num_classes=10)

seed = prep([seed, seed_labels])

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
test_images = (test_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256
BATCH_NUMBER = BUFFER_SIZE//BATCH_SIZE

BATCH_SIZE_VAL = 32

# Batch and shuffle the data
#train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Aufteilen der Trainingsdaten und -labels in batches
train_dataset = [train_images[k*BATCH_SIZE:(k+1)*BATCH_SIZE] for k in range(BATCH_NUMBER)]
train_labels = [train_labels[k*BATCH_SIZE:(k+1)*BATCH_SIZE] for k in range(BATCH_NUMBER)]


log_dir= os.path.join(checkpoint_dir, 'logs')

log_file_train = open(os.path.join(log_dir, 'trainloss_log.txt'), 'w')
log_file_train.write('Epoch, Batch No., Generator Loss Bool, Generator Loss Label, Discriminator Loss Bool, Discriminator Loss Label' + '\n')
#log_file_train.write('Epoch, Batch No., Generator Loss, Discriminator Loss, Accuracy real labels, Accuracy fake labels' + '\n')

log_file_val = open(os.path.join(log_dir, 'val_log.txt'), 'w')
log_file_val.write('Epoch, Label Loss, Label Accuracy, Bool Loss, Bool Accuracy' + '\n')


log_file_misc = open(os.path.join(log_dir, 'miscvalues_log.txt'), 'w')
log_file_misc.write('Epoch, Batch No., Anteil richtiger real labels, Anteil real labels p>0.7, Anteil richtiger fake labels, Anteil fake labels p>0.7' + '\n')
#log_file_misc.write('Epoch, Accuracy real labels, Accuracy fake labels' + '\n')

log_file_trainacc = open(os.path.join(log_dir, 'trainacc_log.txt'), 'w')
log_file_trainacc.write('Epoch, Accuracy real labels, Accuracy fake labels' + '\n')


train(train_dataset, train_labels, EPOCHS)

log_file_train.close()
log_file_val.close()
log_file_misc.close()
log_file_trainacc.close()
#display_image(EPOCHS)
# Laden des zuletzt gespeicherten Modells im Ordner
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


#%%
# noise = seed
# images = train_dataset[0]
# epoch = 0
# batch_counter = 1
# fake_labels = seed_labels
            
# real_labels = to_categorical(train_labels[0], num_classes=10)

# thresh_quant = 0.7
# thresh_prob = 0.2



# a = tf.reduce_sum(tf.cast(tf.math.equal(tf.math.argmax(real_labels, axis = 1), tf.math.argmax(real_labels, axis = 1)), tf.int8))/BATCH_SIZE > thresh_quant and tf.math.reduce_max(real_labels) > thresh_prob

# #a = tf.constant(True)
# b = tf.cond(a, true_fn = lambda:cross_entropy(real_labels, real_labels+1), false_fn =  lambda:2*cross_entropy(real_labels, real_labels+1))

# start1 = time.time()
# with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#     generated_images = generator(noise, training=True)
    
#     real_output = discriminator(images, training=True)
#     fake_output = discriminator(generated_images, training=True)
    
#     real_labels_pred = real_output[:,:-1]
#     fake_labels_pred = fake_output[:,:-1]
        
#     # real_classes_true = np.argmax(real_labels_true, axis = 1)
#     # real_classes_pred = np.argmax(real_labels_pred, axis = 1)
    
#     # log_file_misc.write(str(epoch) + ',' + str(batch_counter) + ',' + 
#     #                     str((np.argmax(real_labels, axis = 1) == np.argmax(real_labels_pred, axis = 1)).sum()/BATCH_SIZE) + ',' + str(np.max(real_labels_pred)) + ',' + 
#     #                     str((np.argmax(fake_labels, axis = 1) == np.argmax(fake_labels_pred, axis = 1)).sum()/BATCH_SIZE) + ',' + str(np.max(fake_labels_pred)) + '\n')

#     # Real Input data
    
#     if (np.argmax(real_labels, axis = 1) == np.argmax(real_labels_pred, axis = 1)).sum()/BATCH_SIZE > thresh_quant and tf.math.reduce_max(real_labels_pred) > thresh_prob:
#         real_loss = cross_entropy(y_true = real_labels, y_pred = real_labels_pred)
#     else:
#         real_loss = 2*cross_entropy(y_true = real_labels, y_pred = real_labels_pred)
    
#     # Fake Input data
#     if (np.argmax(fake_labels, axis = 1) == np.argmax(fake_labels_pred, axis = 1)).sum()/BATCH_SIZE > thresh_quant and tf.math.reduce_max(fake_labels_pred) > thresh_prob:
#         fake_loss = cross_entropy(y_true = fake_labels, y_pred = fake_labels_pred)
#     else:
#         fake_loss = 2*cross_entropy(y_true = fake_labels, y_pred = fake_labels_pred)

#     disc_loss = real_loss + fake_loss
#     gen_loss = fake_loss

# gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
# generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        
# gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
# discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# print(time.time()-start1)


#%%

'''
train_step_labels mit tf.math. Funktionen
'''
# Real Input data
        
# real_cond = tf.reduce_sum(tf.cast(tf.math.equal(tf.math.argmax(real_labels, axis = 1), tf.math.argmax(real_labels_pred, axis = 1)), tf.int8))/BATCH_SIZE > thresh_quant and tf.math.reduce_max(real_labels_pred) > thresh_prob
# real_loss = tf.cond(real_cond, true_fn = lambda:cross_entropy(y_true = real_labels, y_pred = real_labels_pred), false_fn =  lambda:2*cross_entropy(y_true = real_labels, y_pred = real_labels_pred))

# fake_cond = tf.reduce_sum(tf.cast(tf.math.equal(tf.math.argmax(fake_labels, axis = 1), tf.math.argmax(fake_labels_pred, axis = 1)), tf.int8))/BATCH_SIZE > thresh_quant and tf.math.reduce_max(fake_labels_pred) > thresh_prob
# fake_loss = tf.cond(fake_cond, true_fn = lambda:cross_entropy(y_true = fake_labels, y_pred = fake_labels_pred), false_fn =  lambda:2*cross_entropy(y_true = fake_labels, y_pred = fake_labels_pred))

# if tf.reduce_sum(tf.cast(tf.math.equal(tf.math.argmax(real_labels, axis = 1), tf.math.argmax(real_labels_pred, axis = 1)), tf.int8))/BATCH_SIZE > thresh_quant and tf.math.reduce_max(real_labels_pred) > thresh_prob:
#     real_loss = cross_entropy(y_true = real_labels, y_pred = real_labels_pred)
# else:
#     real_loss = 2*cross_entropy(y_true = real_labels, y_pred = real_labels_pred)

# # Fake Input data
# if tf.reduce_sum(tf.cast(tf.math.equal(tf.math.argmax(fake_labels, axis = 1), tf.math.argmax(fake_labels_pred, axis = 1)), tf.int8))/BATCH_SIZE > thresh_quant and tf.math.reduce_max(fake_labels_pred) > thresh_prob:
#     fake_loss = cross_entropy(y_true = fake_labels, y_pred = fake_labels_pred)
# else:
#     fake_loss = 2*cross_entropy(y_true = fake_labels, y_pred = fake_labels_pred)


#%%
