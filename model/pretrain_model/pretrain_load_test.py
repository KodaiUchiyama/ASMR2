import tensorflow as tf
import sys
sys.path.append('../../model/lib/')
import utils
from tensorflow.python.framework import tensor_util
import numpy as np
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

## paraeter
PB = 0
CKPT = 0
HDF5 = 1
#MODEL_PATH = 'FaceNet_keras/facenet_keras.h5'
MODEL_PATH = '/home/kody.uchiyama/keras-facenet/model/facenet_keras.h5'
#input
VALID_FRAME_LOG_PATH = '../../data/video/valid_frame_expanded_test.txt'
FACE_INPUT_PATH = '../../data/video/face_input_expanded_test/'
#output
OUTPUT_TXT_FILE='faceemb_dataset_expanded_test.txt'
SAVE_PATH = './face1022_emb_expanded_test/' 

data = np.random.randint(256,size=(1,160,160,3),dtype='int32')

###############
#graph_path = 'FaceNet_new/20180402-114759.pb'
# utils.inspect_operation(graph_path,'ops.txt')
if PB:
    with tf.gfile.FastGFile(graph_path,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        with tf.Session() as sess:
            #sess.graph.as_default()
            tf.import_graph_def(graph_def)
            print(sess.run('import/embeddings', feed_dict={'import/batch_size:0': data,'import/phase_train:0':False}))

if CKPT:
    saver = tf.train.import_meta_graph('FaceNet_new/model-20180402-114759.meta')
    with tf.Session() as sess:
        saver.restore(sess=sess,save_path='FaceNet_new/model-20180402-114759.ckpt-275')
        #print(sess.run('embeddings:0', feed_dict={'batch_size:0': data, 'phase_train:0': False}))

if HDF5:
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    model = load_model(MODEL_PATH)
    model.summary()
    avgPool_layer_model = Model(inputs=model.input,outputs=model.get_layer('AvgPool').output)
    # print(avgPool_layer_model.predict(data))

    lines = []
    with open(VALID_FRAME_LOG_PATH, 'r') as f:
        lines = f.readlines()

    for line in lines:
        embtmp = np.zeros((75, 1, 1792))
        headname = line.strip() #ex) frame_0-1 from log for face_input
        tailname = ''
        for i in range(1, 76):
            if i < 10:
                tailname = '_0{}.jpg'.format(i)
            else:
                tailname = '_' + str(i) + '.jpg'
            picname = headname + tailname
            # print(picname)
            I = mpimg.imread(FACE_INPUT_PATH + picname)
            I_np = np.array(I)
            #サンプル数用に次元増やす
            I_np = I_np[np.newaxis, :, :, :]
            # print(I_np.shape)
            # print(avgPool_layer_model.predict(I_np).shape)
            embtmp[i - 1, :] = avgPool_layer_model.predict(I_np)

        # print(embtmp.shape)
        frame_index = line.strip().split('_')[1]
        npname = '%s_face_emb.npy'%frame_index #0-0_face_emb.npy
        print(npname)

        np.save(SAVE_PATH + npname, embtmp)
        with open(OUTPUT_TXT_FILE, 'a') as d:
            d.write(npname + '\n')


