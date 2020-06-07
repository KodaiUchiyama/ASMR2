from keras import optimizers
#from keras.layers import Dense, Convolution3D, MaxPooling3D, ZeroPadding3D, Dropout, Flatten, BatchNormalization, ReLU
from keras.models import Sequential, model_from_json
from keras import optimizers
from keras.layers import Input, Dense, Convolution2D, Deconvolution2D, Bidirectional,TimeDistributed
from keras.layers import Dropout, Flatten, BatchNormalization, ReLU, Reshape, Permute, Lambda
from keras.layers.core import Activation
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.layers.recurrent import LSTM
from keras.initializers import he_normal,glorot_uniform
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.callbacks import TensorBoard
import tensorflow as tf
import os
from keras import backend as K
from MyGenerator import DataGenerator



def VO_model():
    def UpSampling2DBilinear(size):
        return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True))
    
    model_input = Input(shape=(75, 1, 1792))
    print('0:', model_input.shape)

    conv1 = Convolution2D(256, kernel_size=(7, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='vs_conv1')(model_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    print('1:', conv1.shape)

    conv2 = Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='vs_conv2')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    print('2:', conv2.shape)

    conv3 = Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(2, 1), name='vs_conv3')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    print('3:', conv3.shape)

    conv4 = Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(4, 1), name='vs_conv4')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    print('4:', conv4.shape)

    conv5 = Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(8, 1), name='vs_conv5')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    print('5:', conv5.shape)

    conv6 = Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(16, 1), name='vs_conv6')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    print('6:', conv6.shape)

    conv7 = Reshape((75, 256, 1))(conv6)
    conv7 = UpSampling2DBilinear((301, 256))(conv7)
    conv7 = Reshape((301, 256))(conv7)
    print('7:', conv7.shape)
    '''
    # --------------------------- VS_model start ---------------------------
    VS_model = Sequential()
    VS_model.add(Convolution2D(256, kernel_size=(7, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='vs_conv1'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='vs_conv2'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(2, 1), name='vs_conv3'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(4, 1), name='vs_conv4'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(8, 1), name='vs_conv5'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Convolution2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(16, 1), name='vs_conv6'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Reshape((75, 256, 1)))
    VS_model.add(UpSampling2DBilinear((298, 256)))
    VS_model.add(Reshape((298, 256)))
    # --------------------------- VS_model end ---------------------------

    video_input = Input(shape=(75, 1, 1792))
    VS_out = VS_model(video_input)
    '''
    AVfusion = TimeDistributed(Flatten())(conv7)
    print('AVfusion:', AVfusion.shape)

    lstm = Bidirectional(LSTM(400, input_shape=(301,256),return_sequences=True),merge_mode='sum')(AVfusion)
    print('lstm:', lstm.shape)
    
    DROPOUT = 0.2
    fc1 = Dense(600, name="fc1", activation='relu', kernel_initializer=he_normal(seed=27))(lstm)
    print('fc1:', fc1.shape)
    fc1 = Dropout(DROPOUT)(fc1)
    
    fc2 = Dense(600, name="fc2", activation='relu', kernel_initializer=he_normal(seed=42))(fc1)
    print('fc2:', fc2.shape)
    fc2 = Dropout(DROPOUT)(fc2)

    fc3 = Dense(600, name="fc3", activation='relu', kernel_initializer=he_normal(seed=65))(fc2)
    print('fc3:', fc3.shape)
    fc3 = Dropout(DROPOUT)(fc3)
    
    complex_mask = Dense(257 * 1, name="complex_mask", kernel_initializer=glorot_uniform(seed=87))(fc3)
    print('complex_mask:', complex_mask.shape)

    complex_mask_out = Reshape((301, 257))(complex_mask)
    model_output = Activation('sigmoid')(complex_mask_out)
    print('model_output:', model_output.shape)

    # --------------------------- VO end ---------------------------
    VO_model = Model(inputs=model_input, outputs=model_output)
    return VO_model


if __name__ == '__main__':
    #############################################################
    RESTORE = True
    # If set true, continue training from last checkpoint
    # needed change 1:h5 file name, 2:epochs num, 3:initial_epoch

    # super parameters
    people_num = 2
    epochs = 50
    initial_epoch = 0
    batch_size = 2
    #############################################################

    # audio_input = np.random.rand(5, 298, 257, 2)        # 5 audio parts, (298, 257, 2) stft feature
    # audio_label = np.random.rand(5, 298, 257, 2, people_num)     # 5 audio parts, (298, 257, 2) stft feature, people num to be defined

    # ///////////////////////////////////////////////////////// #
    # create folder to save models
    path = './saved_models_AO'
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('create folder to save models')
    filepath = path + "/AOmodel-" + str(people_num) + "p-{epoch:03d}-{val_loss:.10f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # checkpoint2 = ModelCheckpoint(path + "/AOmodel-latest-" + str(people_num) + ".h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # ///////////////////////////////////////////////////////// #

    #############################################################
    # automatically change lr
    def scheduler(epoch):
        ini_lr = 0.001
        lr = ini_lr
        if epoch >= 5:
            lr = ini_lr / 5
        if epoch >= 10:
            lr = ini_lr / 10
        return lr

    rlr = LearningRateScheduler(scheduler, verbose=1)
    #############################################################

    # ///////////////////////////////////////////////////////// #
    # read train and val file name
    # format: mix.npy single.npy single.npy
    trainfile = []
    valfile = []
    with open('./trainfile.txt', 'r') as t:
        trainfile = t.readlines()
    with open('./valfile.txt', 'r') as v:
        valfile = v.readlines()
    # ///////////////////////////////////////////////////////// #

    # the training steps
    def latest_file(dir):
        lists = os.listdir(dir)
        lists.sort(key=lambda fn: os.path.getmtime(dir + fn))
        file_latest = os.path.join(dir, lists[-1])
        return file_latest

    if RESTORE:
        last_file = latest_file('./saved_models_AO/')
        AO_model = load_model(last_file)
        info = last_file.strip().split('-')
        initial_epoch = int(info[-2])
        # print(initial_epoch)
    else:
        AO_model = AO_model(people_num)
        adam = optimizers.Adam()
        AO_model.compile(optimizer=adam, loss='mse')
    # AO_model.fit(audio_input, audio_label,
    #              epochs=epochs,
    #              batch_size=2,
    #              validation_data=(audio_input, audio_label),
    #              shuffle=True,
    #              callbacks=[TensorBoard(log_dir='./log_AO'), checkpoint, rlr],
    #              initial_epoch=initial_epoch)

    train_generator = AudioGenerator(trainfile, database_dir_path='./', batch_size=batch_size, shuffle=True)
    val_generator = AudioGenerator(valfile, database_dir_path='./', batch_size=batch_size, shuffle=True)

    AO_model.fit_generator(generator=train_generator,
                           validation_data=val_generator,
                           epochs=epochs,
                           callbacks=[TensorBoard(log_dir='./log_AO'), checkpoint, rlr],
                           initial_epoch=initial_epoch
                           )

