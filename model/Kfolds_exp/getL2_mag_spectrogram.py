import numpy as np
import os, glob
from scipy.spatial.distance import euclidean
#from fastdtw import fastdtw
from matplotlib import pyplot as plt
#import librosa
#import librosa.display # この行が必要
#順序つき辞書
from collections import OrderedDict
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

truesound_file = "../../data/audio/AV_model_database/single/"
predicted_file = "./result/fold4/Predicted_Spectrogram"

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit(p):
    # p-> 0 < p < 1
    ones = np.ones(p.shape)
    p = np.where(p < 0, 0+10**-7, p)
    #ones = ones + 10**-7 これはp=1の時にしか意味がない，かつp=1になることはほぼない
    return np.log(p / (ones - p))

#instead of boxcox, to make the amp distribute more gaussian dist
def log_dist(p):
    return np.log(p + 10**-7)

def log_dist_inv(p):
    return np.exp(p)-10**-7

def mag2db(p):
    return 20 * np.log10(p + 10**-7)

loss_list = []

result_file = 'result_score_mse.txt'

##予測された音声とテスト音声を一つずつ比較
for predicted_path in glob.glob(predicted_file+"/*.npy"):
    #predicted側index抽出
    predicted_id = predicted_path.rsplit('/',1)[1].replace('.npy','')
    
    #各スペクトルを取得
    true_y = np.load(truesound_file + predicted_id + ".npy")
    predicted_y = np.load(predicted_path) # (1, 301, 257)

    #予測スペクトル整形
    #predicted_y = predicted_y
    #print(true_y.shape) # (257, 301)
    #print(predicted_y.shape)
    
    #ノーマライズを解除
    innormalized_true_y = log_dist_inv(logit(true_y))    
    innormalized_predicted_y = log_dist_inv(logit(predicted_y))    
    
    #print(np.max(predicted_y))
    #print(np.min(innormalized_predicted_y))
    #print(np.max(true_y))
    #print(np.min(innormalized_true_y))
    #振幅スペクトルをdBに変換 ydb = 20 log10(y)
    #innormalized_predicted_ydb = mag2db(innormalized_predicted_y)
    #innormalized_true_ydb = mag2db(innormalized_true_y)
    
    #print("------------------------")
    #print("振幅スペクトル（dB）")
    #print(innormalized_predicted_ydb)
    #print(innormalized_true_ydb)
    #print("------------------------")
    # これはMSEでもL2距離でもなく，dB単位の推定誤差として計算している．
    #loss = np.mean(np.abs(innormalized_true_ydb - innormalized_predicted_ydb))
    
    #loss = np.mean(np.abs(innormalized_true_y - innormalized_predicted_y))
    
    #ユークリッド距離：L2 distance np.sqrt(numpy.power(a-b, 2).sum())
    #loss = np.linalg.norm(innormalized_true_y - innormalized_predicted_y)
    loss = np.sqrt(np.power(innormalized_true_y - innormalized_predicted_y, 2).sum())
    #print(innormalized_true_y.shape)
    #print(innormalized_predicted_y.shape)    
    #MSE
    #loss = mean_squared_error(innormalized_true_y, innormalized_predicted_y)
    print("Average error: {0:.5f} ".format(loss))
    loss_list.append(loss)
print(len(loss_list))
loss_list = np.array(loss_list)
print(loss_list)
print("Average of Average error: {0:.2f} ".format(np.mean(loss_list)))
print("Std of Average error: {0:.2f} ".format(np.std(loss_list)))

