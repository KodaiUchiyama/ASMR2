import librosa
import os
import numpy as np
import scipy.io.wavfile as wavfile
import glob

max_num = 0
print("length of list:%d"%len(glob.glob('AV_model_database/single/*.npy')))
for path in glob.glob('AV_model_database/single/*.npy'):
    if (os.path.exists(path)):
        array = np.load(path)
        print(array.max())
        max_num += array.max()
print("average of max: %d"%int(max_num//1037))

#max = 0
#for path in glob.glob('AV_model_database/single/*.npy'):
#    if (os.path.exists(path)):
#        array = np.load(path)
#        max_num = array.max()
#        if (max < max_num):
#            max = max_num
#            print(max_num)
#print(max)
