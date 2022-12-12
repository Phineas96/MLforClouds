#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:24:03 2022

@author: mrosenberger
"""

import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display
import h5py
#import ray

from keras import layers
#from keras.datasets import mnist
from keras.utils import to_categorical

'''
Zweiten train_step für label losses, Entscheidung dafür außerhalb davon,
dann gibt es keine Probleme mit tf.function

Und validation einbauen
'''


# Funktion die den Generator, i.e. den Artist, erstellt
def make_generator_model():
    model = tf.keras.Sequential()
    # model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    # model.add(layers.Reshape((7, 7, 256)))
    # assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, input_shape = [8,8,256]))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(8, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 128, 128, 8)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid'))
    assert model.output_shape == (None, 256, 256, 3)

    return model


# Funktion die den Diskriminator, i.e. den Kritiker, erstellt
# Output des Diskriminators ist <0 wenn fake image und >0 wenn real image
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(8, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[256, 256, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(12, activation = 'sigmoid')) # 11 Kategorien + fake/real

    return model

def make_preparation_model(final_shape= (8,8)):
    in_noise = tf.keras.Input(shape=(100,), name='Generator-Noise-Input-Layer')
    g = layers.Dense(final_shape[0]*final_shape[1]*248, use_bias=False)(in_noise)
    g = layers.BatchNormalization()(g)
    g = layers.LeakyReLU()(g)
    g = layers.Reshape((final_shape[0], final_shape[1], 248))(g)
    
    
    in_label = tf.keras.Input(shape=(11,), name='Generator-Label-Input-Layer') # 11 Kategorien
    lbl = layers.Dense(final_shape[0]*final_shape[1]*8, use_bias=False)(in_label)
    lbl = layers.Reshape((final_shape[0],final_shape[1],8))(lbl)
    
    output = layers.Concatenate(name = 'Concatenation-Layer')([g,lbl])
    
    model = tf.keras.Model([in_noise, in_label], output, name='Preparator')

    return model

# Loss im Diskriminator
# Testergebnis bei echtem Bild mit 1er Vektor und bei falschem Bild mit 0er Vektor verglichen
# Unterteilung des Outputs in bool (letzter Eintrag) und label (die zehn davor)
def discriminator_loss(real_output, real_labels, fake_output, fake_labels):
    real_loss = cross_entropy(tf.ones_like(real_output[:,-1]), real_output[:,-1])
    real_loss_label = cross_entropy(real_labels, real_output[:,:-1])
    
    fake_loss = cross_entropy(tf.zeros_like(fake_output[:,-1]), fake_output[:,-1])
    fake_loss_label = cross_entropy(fake_labels, fake_output[:,:-1])
    
    # lossvec = []
    # for l,o in zip(fake_labels, np.array(fake_output[:,:-1])):
    #     loss = cross_entropy(l,o).numpy()
    #     lossvec.append(loss) if np.argmax(l)==np.argmax(o) and np.max(o)>0.95 else lossvec.append(2*loss)
    
    # fake_loss_label = np.sum(lossvec)
    total_loss = 1.2*(real_loss + fake_loss) + 0.8*(real_loss_label + fake_loss_label)
    return total_loss

# Loss des Generators
# will den Kritiker austricksen, Testergebnis des Diskriminators für fake Bilder wird also mit 1er Vektor verglichen
def generator_loss(fake_output, fake_labels):
    loss_bool = cross_entropy(tf.ones_like(fake_output[:,-1]), fake_output[:,-1])
    loss_label = cross_entropy(fake_labels, fake_output[:,:-1])
    # lossvec = tf.TensorArray(tf.float32, size = BATCH_SIZE)
    # for index in range(fake_labels.shape[0]):
    #     l = fake_labels[index]
    #     o = fake_output[index,:-1]
    #     loss = cross_entropy(l,o)#.numpy()
    #     # tf.print(l)
    #     # tf.print(o)
    #     # tf.print(loss)
    #     if tf.math.argmax(l)==tf.math.argmax(o) and tf.math.reduce_max(o)>0.95:
    #         lossvec = lossvec.write(index, loss)    
    #     else:
    #         lossvec = lossvec.write(index, 2*loss)
    
    # loss_label = tf.reduce_sum(lossvec.stack())
        
    return loss_bool + loss_label

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, real_labels, noise, fake_labels):
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
    
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
    
        gen_loss = generator_loss(fake_output, fake_labels)
        disc_loss = discriminator_loss(real_output, real_labels, fake_output, fake_labels)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    #log_file.write(str(epoch) + ',' + str(batch_counter) + ',' + str(gen_loss) + ',' + str(disc_loss) + '\n')
    
    return gen_loss, disc_loss


def train(dataset, labels, epochs):
    
    for epoch in range(epochs):
        #epoch+=10
        start = time.time()
    
        batch_counter = 1
        for image_batch, label_batch in zip(dataset, labels):
            noise = tf.random.normal([BATCH_SIZE, noise_dim])
            fake_labels = to_categorical(np.random.randint(0, 11, size = BATCH_SIZE), num_classes=11) # 11 Kategorien
            
            # Labels kommen schon one-hot encoded 
            #real_labels = to_categorical(label_batch, num_classes=10)
            real_labels = label_batch
            # ersetzt die ersten Generator Schichten
            noise = prep([noise, fake_labels])
        
            gen_loss, disc_loss = train_step(image_batch, real_labels, noise, fake_labels)
            
            log_file.write(str(epoch) + ',' + str(batch_counter) + ',' + str(gen_loss.numpy()) + ',' + str(disc_loss.numpy()) + '\n')
            
            batch_counter += 1
            
        # display.clear_output(wait=True)
        # generate_and_save_images(generator,
        #                           epoch + 1,
        #                           seed)
        
        # Save the model and a plot every checkpoint_freq epochs
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                  epoch + 1,
                                  seed)
        
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print(gen_loss)
        print(disc_loss)
        
    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                              epochs,
                              seed)
    
    # save final model
    checkpoint.save(file_prefix = checkpoint_prefix)
    
def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])# * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
        plt.title(types[ints[i]])

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

checkpoint_dir = '/jetfs/home/mrosenberger/CNNPictures/CCSN/Models/cGAN'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Details des Trainingsprozesses
EPOCHS = 2000
noise_dim = 100
num_examples_to_generate = 16
checkpoint_freq = EPOCHS//25 # 25 Checkpoints während dem gesamten Trainingsprozess


# prep = tf.keras.Sequential()
# prep.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
# prep.add(layers.BatchNormalization())
# prep.add(layers.LeakyReLU())
# prep.add(layers.Reshape((7, 7, 256)))

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# 11 Kategorien
ints = np.random.randint(0, 11, size = num_examples_to_generate)
seed_labels = to_categorical(ints, num_classes=11)

seed = prep([seed, seed_labels])

types = ['Ac', 'As', 'Cb', 'Cc', 'Ci', 'Cs', 'Ct', 'Cu', 'Ns', 'Sc', 'St']

with h5py.File('/jetfs/home/mrosenberger/CNNPictures/CCSN/Input_256x256.nc', mode='r') as CCSN:
    train_images = CCSN['img_train'][:]
    train_labels = CCSN['label_train'][:]
    img_test = CCSN['img_test'][:]
    label_test = CCSN['label_test'][:]


BUFFER_SIZE = train_images.shape[0]
BATCH_SIZE = 16
BATCH_NUMBER = BUFFER_SIZE//BATCH_SIZE

train_images = train_images.reshape(BUFFER_SIZE, 256, 256, 3).astype('float32')
#train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

images_shuffled = np.ndarray(shape= (BATCH_NUMBER, BATCH_SIZE, 256, 256, 3))
labels_shuffled = np.ndarray(shape= (BATCH_NUMBER, BATCH_SIZE, 11))

for i in range(BATCH_NUMBER):
    indices = np.random.randint(0, BUFFER_SIZE, size = BATCH_SIZE)
    images_shuffled[i] = train_images[indices]
    labels_shuffled[i] = train_labels[indices]
    
# Batch and shuffle the data
#train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Aufteilen der Trainingsdaten und -labels in batches
# train_dataset = [train_images[k*BATCH_SIZE:(k+1)*BATCH_SIZE] for k in range(BATCH_NUMBER)]
# train_labels = [train_labels[k*BATCH_SIZE:(k+1)*BATCH_SIZE] for k in range(BATCH_NUMBER)]


log_dir= os.path.join(checkpoint_dir, 'logs')

log_file = open(os.path.join(log_dir, 'loss_log.txt'), 'w')
log_file.write('Epoch,Batch No.,Generator Loss,Discriminator Loss' + '\n')

train(images_shuffled, labels_shuffled, EPOCHS)

log_file.close()
#%%

pic = np.random.randint(0,len(img_test))

plt.imshow(img_test[pic])
plt.show()
print(label_test[pic])
print(discriminator(tf.expand_dims(img_test[pic], 0)))

picf_seed = tf.random.normal([1, noise_dim])
#picf_int = np.random.randint(0, 11, size = 1)
picf_label = to_categorical(tf.expand_dims(8,0), num_classes=11)

picf_seed = prep([picf_seed, picf_label])

picf = generator(picf_seed)
plt.imshow(picf[0])
plt.show()
print(picf_label)
print(discriminator(picf))

#%%
with open(os.path.join(log_dir, 'loss_log.txt'), 'r') as f:
    reopen = []
    f.readline() # skip header line
    for line in f:
        reopen.append([float(i) for i in line.split(',')[2:]])

        # Falls man mal nicht weiß wie viele Headerlines es sind. Sind dann aber leeren Listen an den Headerpositionen,
        # da gibts sicher einen netten workaround
        # try:
        #     b.append([float(i) for i in line.split(',')[2:]])
        # except:
        #     continue
    
reopen = np.array(reopen)

# get loss of generator/discriminator for each batch in each epoch
g_loss = reopen[:,0]
d_loss = reopen[:,1]

n_batches = len(g_loss)
n_epochs = int(n_batches/BATCH_NUMBER)

g_loss_batchmean = [np.mean(g_loss[k*BATCH_NUMBER:(k+1)*BATCH_NUMBER]) for k in range(n_epochs)]
d_loss_batchmean = [np.mean(d_loss[k*BATCH_NUMBER:(k+1)*BATCH_NUMBER]) for k in range(n_epochs)]

#%%
# Plot them
allax = np.arange(1,n_batches+1)
meanax = np.linspace(BATCH_NUMBER,n_batches,n_epochs)
axlabels = np.linspace(200,n_epochs,n_epochs//200, dtype = 'int32')

plt.figure(figsize = (8,5))
plt.plot(allax, g_loss, alpha = 0.4)
plt.plot(allax, d_loss, alpha = 0.4)
plt.plot(meanax, g_loss_batchmean, c = 'tab:blue', label = 'Generator')
plt.plot(meanax, d_loss_batchmean, c = 'tab:orange', label = 'Discriminator')
plt.legend()
plt.xticks(axlabels*BATCH_NUMBER, labels = axlabels)
plt.xlabel('Epoch No.')
plt.ylabel('Total loss')

plt.savefig(os.path.join(log_dir, 'loss_plot.png'), bbox_inches = 'tight')



