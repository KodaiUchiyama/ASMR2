## evaluate the model and generate the prediction
import sys
sys.path.append('../lib')
from keras.models import load_model
from model_ops import ModelMGPU
import os
import scipy.io.wavfile as wavfile
import numpy as np
import utils
import librosa
import tensorflow as tf
import retrieval_neighbor_v2

import matplotlib
from matplotlib import cm
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# super parameters
people_num = 2
NUM_GPU = 1

# PATH
model_path = './saved_AV_models/AVmodel-14p-025-0.01321.h5'
#model_path = './saved_AV_models/AVmodel-17p-022-0.01356.h5'
#model_path = './saved_AV_models/AVmodel-23p-006-0.01638.h5'

database_path = '../../data/audio/AV_model_database_test/single/'
face_path = '../../data/video/face1022_emb_test/'

# load data should be real test files
testfiles = []
#with open('../../data/AV_log/AVdataset_val_video012.txt', 'r') as f:
with open('../../data/AV_log/AVdataset_test.txt', 'r') as f:
    testfiles = f.readlines()

trainfiles = []
with open('../../data/AV_log/AVdataset_train.txt', 'r') as f:
    trainfiles = f.readlines()


# For predict Spectrogram
def parse_X_data(line, face_path=face_path):
    parts = line.split() # get each name of file for one testset
    #string
    amp_spectrogram = parts[0]
    face_emb = parts[1]
    name = amp_spectrogram.replace('.npy','') #ex)10-49
   
    #making each path 
    amp_spectrogram_path = database_path + amp_spectrogram
    file_path = face_path + face_emb
    #loading each .npy file
    spectrogram_feature = np.load(amp_spectrogram_path)
    face_emb_feature = np.load(file_path)
    #expand dimention, because trained 4 batch size
    face_emb_feature_expand = face_emb_feature[np.newaxis, ...]
    
    return name, face_emb_feature_expand, spectrogram_feature

##### PLOT RESULTS
def plot_spectrogram(name, innormalized_spec, spectrogram_feature):
    plt.subplot(2, 1, 1)
    plt.imshow(innormalized_spec, aspect='auto',vmin=0.0,vmax=1.0)# (pred_idx:end_idx , 42) > (42 , pred_idx:end_idx)
    plt.ylim(0,256)
    plt.title('Predicted feature')
    plt.ylabel('Frequency bands')
    plt.xlabel('Time')
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.imshow(spectrogram_feature, aspect='auto', vmin=-0.0, vmax=1.0)
    plt.ylim(0,256)
    plt.title('Ground Truth')
    plt.ylabel('Frequency bands')
    plt.xlabel('Time')
    plt.colorbar()

    plt.tight_layout()#図の調整
    plt.draw()

    #生成されたスペクトログラムと正解値スペクトログラムを比較表示、保存
    # フォルダの作成
    # make output directory
    #folder = 'Retrieved_feature_fig/'
    folder = 'Predicted_feature_fig/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(folder + "predicted_spectrums-%s.png"%name)  # ../graphs/predicted_spectrums/{lr:0.000597}-{ws:0.000759}/1.png
    plt.close()


# predict data
AV_model = load_model(model_path,custom_objects={"tf": tf})
#print(AV_model.summary())
counter = 0
iteration_num = 500
if NUM_GPU <= 1:
    for line in testfiles:
        print('processing : %s'%line)
        
        #テスト回数
        if counter == iteration_num:
            print('BREAK')
            break

        name, face_emb_feature, spectrogram_feature = parse_X_data(line)
        predicted_spec = AV_model.predict(face_emb_feature)
        '''  
        #予測したスペクトルグラムを保存
        # フォルダの作成
        folder = 'Predicted_Spectrogram'
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save('./%s/%s.npy'%(folder,name),predicted_spec[0,:].T) # (257,301) 
        ''' 
         
        ##A# raw_data, spectrogramを描画
        plot_spectrogram(name, predicted_spec[0,:].T, spectrogram_feature) 
        
        ##B# retrieval_the nearest neighbor, spectrogramを描画
        ##retrieve_neighbor(predicted_spec)
        ##args: predicted spectrogram:shape(301,257)
        #retrieved_spectrogram = retrieval_neighbor_v2.retrieve_neighbor(predicted_spec[0,:]).T #(257,301)
        
        '''
        #retrieveしたスペクトルグラムを保存
        # フォルダの作成
        folder = 'Retrieved_Spectrogram'
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save('./%s/%s.npy'%(folder,name),retrieved_spectrogram) 
        ''' 
        #plot_spectrogram(name,retrieved_spectrogram.T, spectrogram_feature) 
        
        ##A# sigmoidの逆関数logit関数, log(x+10^-7)のガウス的distributionの逆関数log_dist_inv
        innormalized_spec = utils.log_dist_inv(utils.logit(predicted_spec[0,:]))
        
        #B# retrieved_specを元のスケールに戻す
        #innormalized_spec = utils.log_dist_inv(utils.logit(retrieved_spectrogram))
        
        #innormalized_spec = innormalized_spec.T #T-Fの順番をF-Tに修正、griffin limで位相復元のため 
        
        #Grillin Lim phase generatiion 入力はF-T (257,301)にする
        y_inv = librosa.griffinlim(innormalized_spec, hop_length=160, window='hann', center=True) 

        # File for saving predicted wav data
        dir_path = './Predicted_wav/'
        #dir_path = './Retrieved_wav/'
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
       
        filename = dir_path+name+'.wav'
        wavfile.write(filename, 16000, y_inv)
        
        counter += 1

