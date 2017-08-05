from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
import h5py as h5py
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers.core import Dropout
#from keras.utils import plot_model
from scipy import stats
import csv
from keras.callbacks import ModelCheckpoint, History, LambdaCallback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, Dropout
from keras.layers import GlobalMaxPooling2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
import Neurons_Vis




# base_model = VGG19(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
#
# img_path = 'elephant.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# block4_pool_features = model.predict(x)

# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'fc_model.h5'
# dimensions of our images.
img_width, img_height = 224, 224

locpath = "trainingResults"
train_data_dir = 'ULSOrgans_Split/train'
validation_data_dir = 'ULSOrgans_Split/validation'
train_label_list = np.load('train_label_list.npy')
validation_label_list = np.load('validation_label.npy')
nb_train_samples = int(np.sum(train_label_list))
nb_validation_samples = int(np.sum(validation_label_list))
epochs = 50
batch_size = 17
run_in_intel = True
layer_name = 'block5_conv3'

#augmantation parameters
set_rescale = 1./255
set_rotation_range = 0.
set_width_shift_range = 0.
set_height_shift_range = 0.
set_shear_range = 0.
set_zoom_range = 0.
set_horizontal_flip = False
set_vertical_flip = False
set_fill_mode = 'nearest'
set_center = False
set_std = False

print('Data sizes: nb_train_samples')
print(nb_train_samples)
print('nb_validation_samples: ')
print(nb_validation_samples)


#Used to save model and check training progress
def plot_progress(epoch, logs):
    plt.figure()
    plt.plot(range(epoch+1), history.history['loss'], 'b', label='trainin loss')
    plt.plot(range(epoch+1), history.history['val_loss'], 'r', label='validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('training error')
    plt.legend(loc='best')
    plt.savefig(locpath+'training_error.png')
    plt.close('all')


#if run in faculty servers
def create_plot_progress_train(epoch, logs):
    plt.figure()
    plt.plot(range(epoch+1), history.history['loss'], 'b', label='trainin loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('training error')
    plt.legend(loc='best')
    plt.savefig(locpath+'training_error.png')
    plt.close('all')


def create_plot_progress_validation(epoch, logs):
        plt.figure()
        plt.plot(range(epoch + 1), history.history['val_loss'], 'r', label='validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title('training error')
        plt.legend(loc='best')
        plt.savefig(locpath + 'validation_error.png')
        plt.close('all')

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def CreateVGG16(input_shape):
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      include_top=False)
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # x = BatchNormalization(axis=1)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    # x = BatchNormalization(axis=1)(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # x = Dropout(0.2, noise_shape=None, seed=None)(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x = Dropout(0.2, noise_shape=None, seed=None)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = Dropout(0.2, noise_shape=None, seed=None)(x)

#TODO: change this part to the top model we use
    # if include_top:
    #     # Classification block
    #     x = Flatten(name='flatten')(x)
    #     x = Dense(4096, activation='relu', name='fc1')(x)
    #     x = Dense(4096, activation='relu', name='fc2')(x)
    #     x = Dense(classes, activation='softmax', name='predictions')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)

    inputs = img_input
    model = Model(inputs, x, name='vgg16')
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models')

    model.load_weights(weights_path)

    return model


def create_model(IsTrained, IsLoad, weights_to_load=None,layer_name):
    # build the VGG16 network
    # model = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None)  # ,pooling= max)
    model = CreateVGG16((img_width, img_height, 3))
    # model = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling= max)
    print('uploaded trained model')

    # # build a classifier model to put on top of the conv model
    # top_model = Sequential()
    # top_model.add(Flatten(input_shape=(7,7,512)))
    # #top_model.add(Flatten(input_shape=model.output_shape[1:]))
    # top_model.add(Dense(256, activation='relu'))
    # top_model.add(Dropout(0.5))
    # #top_model.add(Dense(256, activation='relu'))
    # #top_model.add(Dense(256, activation='relu'))
    # top_model.add(Dense(18, activation='softmax'))
    #
    # Create your own input format (here 3x224x224)
    #        shape: A shape tuple (integer), not including the batch size.
    #           For instance, `shape=(32,)` indicates that the expected input
    #           will be batches of 32-dimensional vectors.
    #       batch_shape: A shape tuple (integer), including the batch size.
    #           For instance, `batch_shape=(10, 32)` indicates that
    #           the expected input will be batches of 10 32-dimensional vectors.
    #           `batch_shape=(None, 32)` indicates batches of an arbitrary number
    #           of 32-dimensional vectors.
    input = Input(batch_shape=(batch_size, img_width, img_height, 3), shape=(img_width, img_height, 3), name='image_input')

    output_model = model(input)

    top_model = Flatten(name='flatten')(output_model)
    # top_model = Dropout(0.2, noise_shape=None, seed=None)(top_model)
    top_model = Dense(1024, activation='relu', name='fc1')(top_model)
    preds = Dense(18, activation='softmax', name='fc2')(top_model)
    Final_Model = Model(input=input, output=preds)

    # add the model on top of the convolutional base
    # model.add(top_model)


    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = IsTrained

    # Load weights if needed:
    if IsLoad:
        Final_Model.load_weights(weights_to_load)

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    Final_Model.compile(loss='categorical_crossentropy',
                        # optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                        optimizer=optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True),
                        metrics=['accuracy'])
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # show summary and save a graph of the architecture
    Final_Model.summary()
    # plot_model(Final_Model, to_file='model.png', show_layer_names= True, show_shapes=True)

    j = 0
    print('layer number', 'layer name', 'is trainable', 'input shape')
    for layer in Final_Model.layers:
        if layer.name == 'vgg16':
            for vggLayer in layer.layers:
                print(j, vggLayer.name, vggLayer.trainable, vggLayer.input_shape)
                j += 1
        else:
            print(j, layer.name, layer.trainable, layer.input_shape)
        j += 1

    return Final_Model

    #TODO: batch normalization, show prediction visualization,
# necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
#top_model.load_weights(top_model_weights_path)

#TODO: understand nonlinear log-looking dynamic range rescale
# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=None,
    rotation_range=set_rotation_range,
    width_shift_range=set_width_shift_range,
    height_shift_range=set_height_shift_range,
    shear_range=set_shear_range,
    zoom_range=set_zoom_range,
    horizontal_flip=set_horizontal_flip,
    vertical_flip=set_vertical_flip,
    fill_mode=set_fill_mode,
    featurewise_center=set_center,
    featurewise_std_normalization=set_std)

validation_datagen = ImageDataGenerator(
    rescale=None,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=False,
    featurewise_center=set_center,
    featurewise_std_normalization=set_std
)
#

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# features_train = Final_Model.predict_generator(train_generator, nb_train_samples // batch_size)
# np.save(open('features_train.npy', 'w'), features_train)
#
validation_generator = validation_datagen.flow_from_directory(
   validation_data_dir,
   target_size=(img_height, img_width),
   batch_size=batch_size,
   class_mode='categorical')

# features_validation = Final_Model.predict_generator(validation_generator, nb_validation_samples // batch_size)
# np.save(open('features_validation.npy', 'w'), features_validation)

# Create the data:
# train_data = features_train
# temp = [[index]*(train_label_list[index])
#                          for index in range(train_label_list.shape[1])]
# train_labels = np.array([[index]*(train_label_list[index])
#                          for index in range(train_label_list.shape[1])])
# validation_data = features_validation
# validation_labels = np.array([[index] * (validation_label_list[index])
#                          for index in range(validation_label_list.shape[1])])

print('data augmantation : ')
print('rescale = %f, '
      'rotation_range = %f, width_shift_range = %f ,height_shift_range = %f  '
      'shear_range = %df,zoom_range = %f,horizontal_flip = %f,fill_mode = %s'
      % (set_rescale,
         set_rotation_range, set_width_shift_range, set_height_shift_range,
         set_shear_range, set_zoom_range, set_horizontal_flip, set_fill_mode))

if run_in_intel:
    filepath = "weights.bestFirst.hdf5"
    history = History()
    plot_progress1 = LambdaCallback(on_epoch_end=plot_progress)
    check = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1,
                            save_best_only=True, mode='max')
    sample_train_per_batch = nb_train_samples / batch_size
    print('number of batches in train %d' % sample_train_per_batch)
    sample_validatation_per_batch = nb_validation_samples / batch_size
    print('number of batches in validation %d' % sample_validatation_per_batch)

    Final_Model = create_model(False, False)
    # If we run on server
    Results1 = Final_Model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        callbacks=[history, check, plot_progress1]
    )

    print(Results1)

    # create new model, train the top layers and load the weights from the best model of the first model we trained
    Second_Final_Model = create_model(True, True, filepath)

    filepath2 = "weights.bestSecond.hdf5"
    history2 = History()
    plot_progress2 = LambdaCallback(on_epoch_end=plot_progress)
    check2 = ModelCheckpoint(filepath=filepath2, monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')

    print('start train all layers')
    Results2 = Second_Final_Model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        callbacks=[history2, check2, plot_progress2]
    )

    print(Results2)

else:
# checkpoint
    filepath = "weights.bestFirst.hdf5"
    history = History()
    plot_progress_train = LambdaCallback(on_epoch_end=create_plot_progress_train)
    plot_progress_validation = LambdaCallback(on_epoch_end=create_plot_progress_validation)
    check = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1,
                            save_best_only=True, mode='max')
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]

    sample_train_per_batch = nb_train_samples / batch_size
    print('number of batches in train %d' %sample_train_per_batch)
    # print('sample_train_per_batch: %d') % sample_train_per_batch
    sample_validatation_per_batch = nb_validation_samples / batch_size
    print('number of batches in validation %d' %sample_validatation_per_batch)
    # print('sample_validatation_per_batch: %d') % sample_validatation_per_batch

    # If we run locally
    # Final_Model = create_model(False, False)
    # Results1 = Final_Model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=sample_train_per_batch,
    #     epochs=epochs,
    #     validation_data=validation_generator,
    #     validation_steps=sample_validatation_per_batch
    # )

    # # show summary and save a graph of the architecture
    # print(Results1)
    # Second_Final_Model = create_model(True, True, filepath)
    # print('start train all layers')
    #
    # Results2 = Second_Final_Model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=nb_train_samples,
    #     nb_epoch=epochs,
    #     validation_data=validation_generator,
    #     validation_steps=nb_validation_samples
    # )
    #
    # print(Results2)

    # Build the model without train the top layers:
    Final_Model = create_model(False, False)
    # If we run on server
    Results1 = Final_Model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=epochs,
        # validation_data=validation_generator,
        # nb_val_samples=nb_validation_samples,
        callbacks=[history, check, plot_progress_train]
    )

    ValResults1 = Final_Model.evaluate_generator(
        validation_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=epochs,
        # validation_data=validation_generator,
        # nb_val_samples=nb_validation_samples,
        callbacks=[history, check, plot_progress_validation]
    )

    print(Results1)

    # create new model, train the top layers and load the weights from the best model of the first model we trained
    Second_Final_Model = create_model(True, True, filepath)

    filepath2 = "weights.bestSecond.hdf5"
    history2 = History()
    plot_progress_train2 = LambdaCallback(on_epoch_end=create_plot_progress_train)
    plot_progress_validation2 = LambdaCallback(on_epoch_end=create_plot_progress_validation)
    check2 = ModelCheckpoint(filepath=filepath2, monitor='val_acc', verbose=1,
                            save_best_only=True, mode='max')

    print('start train all layers')
    Results2 = Second_Final_Model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=epochs,
            # validation_data=validation_generator,
            # nb_val_samples=nb_validation_samples,
            callbacks=[history2, check2, plot_progress_train2]
    )

    ValResults2 = Second_Final_Model.evaluate_generator(
        validation_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=epochs,
        # validation_data=validation_generator,
        # nb_val_samples=nb_validation_samples,
        callbacks=[history2, check2, plot_progress_validation2]
    )
    
    # visualize 64 strongest neurons of layer_name
    Neurons_Vis.visualize_neuron(layer_name)
    print(Results2)

# If we run on server- split train and validate
# Results = Final_Model.fit_generator(
#     train_generator,
#     samples_per_epoch=sample_train_per_batch,
#     nb_epoch=epochs,
# )
#
# Final_Model.evaluate_generator(
#     generator=validation_generator,
#     val_samples=sample_validatation_per_batch
# )


# History = Final_Model.fit(train_data, train_labels, batch_size=13, nb_epoch=30, callbacks=callbacks_list,
#                           verbose=1, validation_data=(validation_data, validation_labels))
# print(History)
#
#scalar_test_loss = Final_Model.evaluate(validation_generator, batch_size=sample_validatation_per_batch, verbose=1,
#                                        sample_weight=None)

Final_Model.save_weights(top_model_weights_path)
#
#print('model.metrics_names:')
#print(Final_Model.metrics_names)
#
#print('scalar test loss:')
#print(scalar_test_loss)


## fine-tune the model
# Final_Model.fit_generator(
#     train_generator,
#     samples_per_epoch=nb_train_samples,
#     epochs=epochs,
#     validation_data=validation_generator,
#     nb_val_samples=nb_validation_samples)




# X_train_after = np.load('X_train_after.npy')
# Y_train_after = np.load('Y_train_after.npy')
# X_train_after_transpose = np.transpose(X_train_after,[0,3,2,1])
#
# X_test_after = np.load('X_test_after.npy')
# Y_test_after = np.load('Y_test_after.npy')
# X_test_after_transpose = np.transpose(X_test_after,[0,3,2,1])
#Y_train_after_transpose = np.transpose(Y_train_after)

# X_test_after_transpose_matched = X_test_after_transpose[:-3, :, :, :]
# Y_test_after_matched = Y_test_after[:-3, :]

#predictions = Final_Model.predict(X_train_after_transpose, batch_size= 16)
#print('predictions:')
#print(predictions)

# train_loss=Results2.history['loss']
# val_loss=Results2.history['val_loss']
# train_acc=Results2.history['acc']
# val_acc=Results2.history['val_acc']
# xc=range(epochs)
#
# plt.figure(1, figsize=(7,5))
# plt.plot(xc, train_loss)
# plt.plot(xc, val_loss)
# plt.xlabel('num of Epochs')
# plt.ylabel('loss')
# plt.title('train_loss vs val_loss')
# plt.grid(True)
# plt.legend(['train', 'val'])
# print(plt.style.available) # use bmh, classic,ggplot for big pictures
# plt.style.use(['classic'])
#
# plt.figure(2, figsize=(7,5))
# plt.plot(xc, train_acc)
# plt.plot(xc, val_acc)
# plt.xlabel('num of Epochs')
# plt.ylabel('accuracy')
# plt.title('train_acc vs val_acc')
# plt.grid(True)
# plt.legend(['train', 'val'],loc=4)
# #print plt.style.available # use bmh, classic,ggplot for big pictures
# plt.style.use(['classic'])
