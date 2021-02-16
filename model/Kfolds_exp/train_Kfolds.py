########################
# テストデータを拡張後，グループK分割交差検証で実験を行う
'''
データセット
fold1
AVdataset_fold1_train.txt
AVdataset_fold1_test.txt
fold2
AVdataset_fold2_train.txt
AVdataset_fold2_test.txt
fold3
AVdataset_fold3_train.txt
AVdataset_fold3_test.txt
fold4
AVdataset_fold4_train.txt
AVdataset_fold4_test.txt
###
'''
import sys
sys.path.append('../lib')
#import model_AV_new as AV
import model_VO_amp as VO
from model_ops import ModelMGPU,latest_file
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, CSVLogger
from keras.models import Model, load_model
from MyGenerator import DataGenerator
from MyGenerator import DataGenerator_highFreq
from keras.callbacks import TensorBoard
from keras import optimizers
import os
from model_loss import audio_discriminate_loss2 as audio_loss
import tensorflow as tf
import random
import math

# create AV model
#############################################################
RESTORE = False
# If set true, continue training from last checkpoint
# needed change 1:h5 file name, 2:epochs num, 3:initial_epoch

# super parameters
experiment_ver = 4
epochs = 100
initial_epoch = 0
batch_size = 4 # 4 to feed one 16G GPU
gamma_loss = 0.1
beta_loss = gamma_loss*2

# physical devices option to accelerate training process
workers = 1 # num of core
use_multiprocessing = False
NUM_GPU = 1

# PATH
path = './saved_Kfolds_models' # model path
path_log = './training_log' # log path
database_dir_path = '../../data/'
#############################################################

# create folder to save models
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)
    print('create folder to save models')
filepath = path + "/AVmodel-" + str(experiment_ver) + "p-{epoch:03d}-{val_loss:.5f}.h5"
# period=1 オプション1epoch ごとに保存する
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


#############################################################
# automatically change lr
def scheduler(epoch):
    #ini_lr = 0.00001 #adam original
    ini_lr = 0.001 #adam
    #ini_lr = 0.01 #sgd
    lr = ini_lr
    if epoch >= 5:
        lr = ini_lr / 1
    if epoch >= 10:
        lr = ini_lr / 1
    return lr

rlr = LearningRateScheduler(scheduler, verbose=1)

# create folder to save logs
folder = os.path.exists(path_log)
if not folder:
    os.makedirs(path_log)
    print('create folder to save logs')

csv_logger = CSVLogger(path_log+'/training-'+ str(experiment_ver) +'.log',separator=',',append=False)
#############################################################
# read train and val file name
# format: mix.npy single.npy single.npy
trainfile = []
valfile = []
with open((database_dir_path+'AV_log/AVdataset_fold4_train.txt'), 'r') as t:
    trainfile = t.readlines()
    # ランダムシャッフルして，1割を検証データとする 
    random.seed(7)
    test_ratio = 0.1
    random.shuffle(trainfile)

    length = len(trainfile)
    mid = int(math.floor(test_ratio*length))
    val_data = trainfile[:mid]
    train_data = trainfile[mid:]
# ///////////////////////////////////////////////////////// #

# the training steps
if RESTORE:
    latest_file = latest_file(path+'/')
    AV_model = load_model(latest_file,custom_objects={"tf": tf})
    info = latest_file.strip().split('-')
    initial_epoch = int(info[-2])
else:
    VO_model = VO.VO_model()
    
train_generator = DataGenerator(train_data,database_dir_path= database_dir_path, batch_size=batch_size, shuffle=True)
val_generator = DataGenerator(val_data,database_dir_path=database_dir_path, batch_size=batch_size, shuffle=True)

if NUM_GPU <= 1:
    adam = optimizers.Adam()
    loss = 'mean_squared_error'
    VO_model.compile(optimizer=adam, loss=loss)
    print(VO_model.summary())
    VO_model.fit_generator(generator=train_generator,
                           validation_data=val_generator,
                           epochs=epochs,
                           workers = workers,
                           use_multiprocessing= use_multiprocessing,
                           callbacks=[TensorBoard(log_dir='./log_AV'), checkpoint, rlr, csv_logger],
                           initial_epoch=initial_epoch
                           )

