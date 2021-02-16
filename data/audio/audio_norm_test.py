import librosa
import os
import numpy as np
import scipy.io.wavfile as wavfile
import glob

RANGE = (0,10)
output_path = 'norm_audio_expanded_test_index_35-44'
if(not os.path.isdir(output_path)):
    os.mkdir(output_path)

for num in range(RANGE[0],RANGE[1]):
    # "audio_train{num}"で始まるファイルのみを取得
    for path in glob.glob('audio_expanded_test/audio_test%s-*.wav'%num):
        #normalized file name
        segment_num = path.split('-')[1].replace('.wav','')
        #indexを0-9から35-44にするため，+35でファイル名を書き出し
        norm_path = output_path + '/trim_audio_train%s-%s.wav'%(num+35,segment_num)
        if (os.path.exists(path)):
            audio,_= librosa.load(path,sr=16000)
            max = np.max(np.abs(audio))
            norm_audio = np.divide(audio,max)
            wavfile.write(norm_path,16000,norm_audio)

















