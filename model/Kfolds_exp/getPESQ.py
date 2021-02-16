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
from scipy.io import wavfile
from pesq import pesq

# 正解wavファイル
truesound_file = '/home/kody.uchiyama/speech_separation-master/data/audio/norm_audio_train/' #ex)trim_audio_train0-0.wav expanded_testのindexは35-44に変更され追加済み

# 予測したwavファイル
predicted_file = "./result/fold4/Predicted_wav/" 
#predicted_file = ["./Predicted_wav/","Predicted_wav_"] 
#predicted_file = ["./Retrieved_wav/","Retrieved_wav_"] 

pesq_list = []

#result_file = 'result_score_mse.txt'

##予測された音声とテスト音声を一つずつ取得
for predicted_path in glob.glob(predicted_file+"*.wav"):
    #predicted側index抽出
    predicted_id = predicted_path.rsplit('/',1)[1].replace('.wav','') #1-16
    
    #正解音声を取得 16kHz
    rate_ref, ref = wavfile.read(truesound_file + "trim_audio_train" + predicted_id + ".wav")
    #予測した音声を取得 16kHz
    rate_deg, deg = wavfile.read(predicted_path)
    #print(rate_ref)
    #print(rate_deg)
    #print(ref.shape)
    #print(deg.shape)
    output_pesq = pesq(rate_ref, ref, deg, 'wb')
    # PESQが出力されないときがある-7など，そのときは無視
    if output_pesq < 0:
        print("processing error")
        continue
        '''
        SUCCESS                =  0
        UNKNOWN                = -1
        INVALID_SAMPLE_RATE    = -2
        OUT_OF_MEMORY_REF      = -3
        OUT_OF_MEMORY_DEG      = -4
        OUT_OF_MEMORY_TMP      = -5
        BUFFER_TOO_SHORT       = -6
        NO_UTTERANCES_DETECTED = -7
        '''

    # PESQ計算
    print("PESQ:"+ predicted_id+": "+ str(output_pesq))
    
    pesq_list.append(output_pesq)
print("List length:"+str(len(pesq_list)))
pesq_list = np.array(pesq_list)
print(pesq_list)
print("Average of PESQ:"+str(np.mean(pesq_list)))

