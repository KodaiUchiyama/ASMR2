#Retrive the nearest neighbor spectrogram from training dataset.
#argumet predicted spectrogram
#return retrived spectrogram (300,257) array
import sys
import os
import numpy as np

# PATH
database_path = '../../data/audio/AV_model_database/single/'

# load data
trainfiles = []
with open('../../data/AV_log/AVdataset_val.txt', 'r') as f:
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

#args same size array
#return -1 - +1 similarlity
def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

#args: predicted spectrogram shape(301,257) 
#return: retrieved spectrogram shape(301,257) 
def retrieve_neighbor(predicted_spec):
    #空スペクトルグラム初期化 
    #print(predicted_spec.shape)
    retrieved_spectrogram = np.zeros((1,257))
    for pred_segment in np.array_split(predicted_spec, 12): #12iter
        #print(pred_segment.shape)
        cos_sim = -1
        #trainfile_segmentは(25,257)(26,257)が混在するため、大きさを揃える
        nearest_segment = np.zeros((25,257))
        for line in trainfiles:                             #videos * segments * 12 iter
            name, spectrogram = parse_Y_data(line) # spectrogram.shape = (257,301)
            #print("%s, %s"%(name,spectrogram.shape))
            for trainfile_segment in np.array_split(spectrogram.T, 12): #12iter
                #print(trainfile_segment.shape)
                #print(pred_segment.shape)
                cos_sim_new = cosine_sim(np.mean(pred_segment, axis=0), np.mean(trainfile_segment, axis=0)) #mean time dimention (25,257)->(257,) 
                #print(cos_sim_new)
                #類似度が高かったら、nearest neighborを更新
                if cos_sim_new > cos_sim:
                    nearest_segment = trainfile_segment[0:25,:] #shape(25,257)のみ抜粋
                    cos_sim = cos_sim_new
        
        retrieved_spectrogram = np.concatenate([retrieved_spectrogram, nearest_segment])
        print("nearest_cos_simi:%s"%cos_sim)
        #print(retrieved_spectrogram.shape)
    return retrieved_spectrogram

#test 
#a = np.load(database_path + "2-5.npy").T

#print(a.shape)
#print(np.mean(a,axis=0).shape)
#retrieved_spectrogram = retrieve_neighbor(a)
#print(retrieved_spectrogram.shape)
