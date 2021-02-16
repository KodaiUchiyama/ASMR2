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
import retrieval_neighbor_v2# retrieval 25frames
import retrieval_neighbor_v3# retrieval wav direct
import retrieval_neighbor_v4 # retrieval 10frames

import matplotlib
from matplotlib import cm
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# super parameters
people_num = 2
NUM_GPU = 1

# model PATH
#fold1
#model_path = './saved_Kfolds_models/AVmodel-1p-022-0.01333.h5'
#fold2
#model_path = './saved_Kfolds_models/AVmodel-2p-012-0.01378.h5'
#fold3
#model_path = './saved_Kfolds_models/AVmodel-3p-016-0.01468.h5'
#fold4
model_path = './saved_Kfolds_models/AVmodel-4p-014-0.01533.h5'

database_path = '../../data/audio/AV_model_database/single/'
face_path = '../../data/video/face1022_emb/'
database_dir_path = '../../data/'

# load data should be real test files
testfiles = []
with open((database_dir_path+'AV_log/AVdataset_fold4_test.txt'), 'r') as f:
    testfiles = f.readlines()

trainfiles = []
with open((database_dir_path+'AV_log/AVdataset_fold4_train.txt'), 'r') as f:
    trainfiles = f.readlines()


# 入力データの取得
def parse_X_data(line, face_path=face_path):
    # 35-6.npy 35-6_face_emb.npy
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
    folder = 'Retrieved_feature_fig/'
    #folder = 'Predicted_feature_fig/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(folder + "predicted_spectrums-%s.png"%name)  # ../graphs/predicted_spectrums/{lr:0.000597}-{ws:0.000759}/1.png
    plt.close()


# predict data
AV_model = load_model(model_path,custom_objects={"tf": tf})
#print(AV_model.summary())
counter = 0
#iteration_num = 10
length = len(testfiles)
print("testfiles length:"+str(length))
if NUM_GPU <= 1:
    for line in testfiles:
        #print('processing : %s'%line)
        if counter % 100 == 0:
            print('file counter:'+str(counter))
        '''
        #テスト回数
        if counter == iteration_num:
            print('BREAK')
            break
        ''' 
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
        #plot_spectrogram(name, predicted_spec[0,:].T, spectrogram_feature) 
        
        # fold1のみ生成，主観評価のために動画データが必要であるが，動画データはfold1のテストデータしかないから
        ##B# retrieval the nearest neighbor, spectrogramとwavを取得
        ##args: predicted spectrogram:shape(301,257), trainfiles
        # v3ではスペクトログラムだけでなく, wavファイルを取得する
        #retrieved_spectrogram, retrieved_wav = retrieval_neighbor_v3.retrieve_neighbor(predicted_spec[0,:], trainfiles) #(257,301),(48000,)
        # v2 
        # retrieved_spectrogram = retrieval_neighbor_v2.retrieve_neighbor(predicted_spec[0,:], trainfiles) #(257,301),(48000,)
        
        ''' PESQでも，L2距離でも使わない
        #retrieveしたスペクトルグラムを保存
        # フォルダの作成
        folder = 'Retrieved_Spectrogram'
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save('./%s/%s.npy'%(folder,name),retrieved_spectrogram) 
        '''
        ##B#retrivedしたスペクトログラムを図で保存
        #plot_spectrogram(name,retrieved_spectrogram, spectrogram_feature) 
        
       
       
        #A# sigmoidの逆関数logit関数, log(x+10^-7)のガウス的distributionの逆関数log_dist_inv
        innormalized_spec = utils.log_dist_inv(utils.logit(predicted_spec[0,:]))
        
        #B# retrieved_specを元のスケールに戻す
        #innormalized_spec = utils.log_dist_inv(utils.logit(retrieved_spectrogram))
        
        
        
        innormalized_spec = innormalized_spec.T #T-Fの順番をF-Tに修正、griffin limで位相復元のため 
        #Grillin Lim phase generatiion 入力はF-T (257,301)にする
        y_inv = librosa.griffinlim(innormalized_spec, hop_length=160, window='hann', center=True) 
        # File for saving predicted wav data
        dir_path = './Predicted_wav/'
        #dir_path = './Retrieved_wav/'
        #dir_path = './Retrieved_wav_griffin/'
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
       
        filename = dir_path+name+'.wav'
        wavfile.write(filename, 16000, y_inv)
        #wavfile.write(filename, 16000, retrieved_wav)
        counter += 1
