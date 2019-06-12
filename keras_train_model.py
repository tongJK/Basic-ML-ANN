# SinisterJK
# https://github.com/tongJK/Basic-ML-ANN/edit/master/keras_train_model.py

import os
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

from resnet_152 import resnet152_model

verbose = 1
patience = 50
batch_size = 16
num_channels = 3
img_width, img_height = 224, 224 # size of train & validation images

train_data = 'train' # path of train data
validation_data = 'validation' # path of validate data

num_classes = 224 # number of data classes
num_train_samples = 9985 # number of test samples
num_valid_samples = 2126 # number of validate samples

num_epochs = 100 # number of ahh, in my language is how many round you want to your trained model to evolution

if __name__ == '__main__':
    # build a classifier model
    model = resnet152_model(img_height, img_width, num_channels, num_classes)

    # prepare data augmentation configuration
    train_data_gen = ImageDataGenerator(rotation_range=20., width_shift_range=0.1, height_shift_range=0.1,
                                        zoom_range=0.2, rescale=1. / 255, horizontal_flip=True)
    valid_data_gen = ImageDataGenerator()

    # callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    log_file_path = 'logs/training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1, patience=int(patience / 4), verbose=1)

    trained_models_path = 'models/'
    model_names = os.path.join(trained_models_path, 'model.{epoch:02d}_with_{val_acc:.2f}.hdf5')

    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
    callbacks = [tensor_board, model_checkpoint, csv_logger, early_stop, reduce_lr]

    # generators
    train_generator = train_data_gen.flow_from_directory(train_data, (img_width, img_height), batch_size=batch_size,
                                                         class_mode='categorical')
    valid_generator = valid_data_gen.flow_from_directory(validation_data, (img_width, img_height),
                                                         batch_size=batch_size, class_mode='categorical')

    # fine tune the model
    model.fit_generator(train_generator, steps_per_epoch=num_train_samples / batch_size,
                        validation_data=valid_generator, validation_steps=num_valid_samples / batch_size,
                        epochs=num_epochs, callbacks=callbacks, verbose=verbose)
