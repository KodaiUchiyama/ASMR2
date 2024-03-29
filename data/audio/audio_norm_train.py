import librosa
import os
import numpy as np
import scipy.io.wavfile as wavfile
import glob

RANGE = (0,35)

if(not os.path.isdir('norm_audio_train')):
    os.mkdir('norm_audio_train')

for num in range(RANGE[0],RANGE[1]):
    # "audio_train{num}"で始まるファイルのみを取得
    #print(glob.glob('audio_train/audio_train%d-*.wav'%num))
    for path in glob.glob('audio_train/audio_train%s-*.wav'%num):
        #normalized file name
        segment_num = path.split('-')[1].replace('.wav','')
        norm_path = 'norm_audio_train/trim_audio_train%s-%s.wav'%(num,segment_num)
        if (os.path.exists(path)):
            audio,_= librosa.load(path,sr=16000)
            max = np.max(np.abs(audio))
            norm_audio = np.divide(audio,max)
            wavfile.write(norm_path,16000,norm_audio)

















