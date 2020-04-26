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

import matplotlib
from matplotlib import cm
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# super parameters
people_num = 2
NUM_GPU = 1

# PATH
model_path = './saved_AV_models/AVmodel-14p-025-0.01321.h5'
#model_path = './saved_AV_models/AVmodel-15p-011-0.06611.h5'
dir_path = './pred/'
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)
database_path = '../../data/audio/AV_model_database/single/'
face_path = '../../data/video/face1022_emb/'

# load data
testfiles = []
with open('../../data/AV_log/AVdataset_val.txt', 'r') as f:
    testfiles = f.readlines()

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
    spectrogram_feature_expand = spectrogram_feature[np.newaxis, ...]
    face_emb_feature_expand = face_emb_feature[np.newaxis, ...]
    
    return name, face_emb_feature_expand, spectrogram_feature_expand

##### PLOT RESULTS
def plot_spectrogram(name, innormalized_spec, spectrogram_feature):
    plt.subplot(2, 1, 1)
    plt.imshow(innormalized_spec, aspect='auto',vmin=0.0,vmax=1.0)# (pred_idx:end_idx , 42) > (42 , pred_idx:end_idx)
    plt.title('Predicted feature')
    plt.ylabel('Frequency bands')
    plt.xlabel('Time (frames)')
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.imshow(spectrogram_feature, aspect='auto', vmin=-0.0, vmax=1.0)
    plt.title('Ground Truth')
    plt.ylabel('Frequency bands')
    plt.xlabel('Time (frames)')
    plt.colorbar()

    plt.tight_layout()#図の調整
    plt.draw()

    #生成されたスペクトログラムと正解値スペクトログラムを比較表示、保存
    # フォルダの作成
    # make output directory
    folder = 'Predicted_feature/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(folder + "predicted_spectrums-%s.png"%name)  # ../graphs/predicted_spectrums/{lr:0.000597}-{ws:0.000759}/1.png
    plt.close()


# predict data
AV_model = load_model(model_path,custom_objects={"tf": tf})
#print(AV_model.summary())
counter = 0
iteration_num = 50
if NUM_GPU <= 1:
    for line in testfiles:
        name, face_emb_feature, spectrogram_feature = parse_X_data(line)
        predicted_spec = AV_model.predict(face_emb_feature)
        
        print("sigmoid_array")
        print("sigmoid_array max")
        print(predicted_spec.max())
        print("sigmoid_array min")
        print(predicted_spec.min())
        
        #spectrogramを描画
        plot_spectrogram(name,predicted_spec[0,:].T, spectrogram_feature[0,:]) 
        
        #sigmoidの逆関数logit関数, log(x+10^-7)のガウス的distributionの逆関数log_dist_inv
        innormalized_spec = utils.log_dist_inv(utils.logit(predicted_spec[0,:]))
        #振幅スペクトルに負の値は存在しないから。logit関数の結果から、値が負になるものは0パディングする。（これはoutput layer sigmoidを追加することによって必要なし
        #innormalized_spec = np.where(innormalized_spec < 0, 0, innormalized_spec)
        
        print("logit_array")
        print("logit_array max")
        print(innormalized_spec.max())
        print("logit_array min")
        print(innormalized_spec.min())
        innormalized_spec = innormalized_spec.T #T-Fの順番をF-Tに修正、griffin limで位相復元のため 
        
        #Grillin Lim phase generatiion
        y_inv = librosa.griffinlim(innormalized_spec, hop_length=160, window='hann', center=True) 
        
        filename = dir_path+name+'.wav'

        #テスト回数
        wavfile.write(filename, 16000, y_inv)
        if counter == iteration_num:
            print('BREAK')
            break
        counter += 1
#if NUM_GPU > 1:
#    parallel_model = ModelMGPU(AV_model,NUM_GPU)
#    for line in testfiles:
#        mix,single_idxs,face_embs = parse_X_data(line)
#        mix_expand = np.expand_dims(mix, axis=0)
#        cRMs = parallel_model.predict([mix_expand,face_embs])
#        cRMs = cRMs[0]
#        prefix = ""
#        for idx in single_idxs:
#            prefix += idx + "-"
#        for i in range(len(cRMs)):
#            cRM = cRMs[:,:,:,i]
#            assert cRM.shape == (298,257,2)
#            F = utils.fast_icRM(mix,cRM)
#            T = utils.fast_istft(F,power=False)
#            filename = dir_path+prefix+str(single_idxs[i])+'.wav'
#            wavfile.write(filename,16000,T)




