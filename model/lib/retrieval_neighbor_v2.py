#Retrive the nearest neighbor spectrogram(L2 distance version) from training dataset.
#argumet predicted spectrogram
#return retrived spectrogram (300,257) array
import sys
import os
import numpy as np

# PATH
database_path = '../../data/audio/AV_model_database/single/'

# load data
trainfiles = []
with open('../../data/AV_log/AVdataset_train.txt', 'r') as f:
    trainfiles = f.readlines()

# For predict Spectrogram
def parse_Y_data(line):
    parts = line.split() # get each name of file for one testset
    #string
    amp_spectrogram = parts[0]
    face_emb = parts[1]
    name = amp_spectrogram.replace('.npy','') #ex)10-49
   
    #making each path 
    amp_spectrogram_path = database_path + amp_spectrogram
    #file_path = face_path + face_emb
    #loading each .npy file
    spectrogram_feature = np.load(amp_spectrogram_path)
    #print(spectrogram_feature.shape)
    #face_emb_feature = np.load(file_path)
    #expand dimention, because trained 4 batch size
    #spectrogram_feature_expand = spectrogram_feature[np.newaxis, ...]
    #face_emb_feature_expand = face_emb_feature[np.newaxis, ...]
    
    return name, spectrogram_feature

#args same size 2d array
#L2 disctance
def euclidean_dist(v1, v2):
    return np.linalg.norm(v1 - v2)

#args: predicted spectrogram shape(301,257) 
#return: retrieved spectrogram shape(301,257) 
def retrieve_neighbor(predicted_spec):
    #空スペクトルグラム初期化 
    #print(predicted_spec.shape)
    retrieved_spectrogram = np.zeros((1,257))
    for pred_segment in np.array_split(predicted_spec, 12): #12iter
        #print(pred_segment.shape)
        dist_init_flag = True
        #trainfile_segmentは(25,257)(26,257)が混在するため、大きさを揃える
        nearest_segment = np.zeros((25,257))
        for line in trainfiles:                             #videos * segments * 12 iter
            name, spectrogram = parse_Y_data(line) # spectrogram.shape = (257,301)
            #print("%s, %s"%(name,spectrogram.shape))
            for trainfile_segment in np.array_split(spectrogram.T, 12): #12iter
                #print(trainfile_segment.shape)
                #print(pred_segment.shape)
                #(301,257)を12等分したら(26,257)が混ざるから、初めの25列のみで距離を計算
                dist_new = euclidean_dist(pred_segment[0:25,:], trainfile_segment[0:25,:])
                #print(dist_new)
                if dist_init_flag:
                    dist = dist_new
                    dist_init_flag = False
                #L2距離が小さかったら、nearest neighborを更新
                if dist_new <= dist:
                    nearest_segment = trainfile_segment[0:25,:] #shape(25,257)のみ抜粋
                    dist = dist_new
        
        retrieved_spectrogram = np.concatenate([retrieved_spectrogram, nearest_segment])
        print("nearest_dist:%s"%dist)
        #print(retrieved_spectrogram.shape)
    return retrieved_spectrogram

#test 
#a = np.load(database_path + "2-5.npy").T

#print(a.shape)
#print(np.mean(a,axis=0).shape)
#retrieved_spectrogram = retrieve_neighbor(a)
#print(retrieved_spectrogram.shape)