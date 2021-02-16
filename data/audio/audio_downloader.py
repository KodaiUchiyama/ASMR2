# Before running, make sure avspeech_train.csv and avspeech_test.csv are in catalog.
# if not, see the requirement.txt
# download and preprocess the data from AVspeech dataset
import os
import sys
sys.path.append("../lib")
import AVHandler as avh
import pandas as pd

def m_link(youtube_id):
    # return the youtube actual link
    link = 'https://www.youtube.com/watch?v='+youtube_id
    return link

def m_audio(loc,name,cat,start_idx,end_idx):
    # make concatenated audio following by the catalog from AVSpeech
    # loc       | the location for file to store
    # name      | name for the wav mix file
    # cat       | the catalog with audio link and time
    # start_idx | the starting index of the audio to download and concatenate
    # end_idx   | the ending index of the audio to download and concatenate
    
    #データセットRange(0-end)
    for i in range(start_idx,end_idx):
        f_name = name+str(i)
        link = m_link(cat.loc[i,'link'])
        start_time = cat.loc[i,'start_time']
        end_time = cat.loc[i,'end_time']
        start_time_sec = int(start_time.strip().split("-")[0])*60+int(start_time.strip().split("-")[1])
        end_time_sec = int(end_time.strip().split("-")[0])*60+int(end_time.strip().split("-")[1])
        print('index_num:%s'%i) 
        
        win_num =int((end_time_sec - start_time_sec) / 3)
        print('segment_size:%s' % win_num)
        
        #16000 down sampling rate wav file download
        avh.download(loc,f_name,link)
        
        for j in range(win_num):
            segment_f_name = f_name + "-" + str(j)
            avh.cut(loc,f_name,segment_f_name,start_time_sec)
            #step size 3s
            start_time_sec = start_time_sec + 3
        #delete original audio file
        command = 'cd %s;' % loc
        command += 'rm %s.wav' % f_name
        os.system(command)

#dataset csv file
#cat_train = pd.read_csv('catalog/avspeech_train.csv')
cat_test = pd.read_csv('catalog/avspeech_expanded_test.csv')

# create 80000-90000 audios data from 290K
#avh.mkdir('audio_train')
#avh.mkdir('audio_test')
avh.mkdir('audio_expanded_test')
#m_audio('audio_train','audio_train',cat_train,0,35)
#m_audio('audio_test','audio_test',cat_test,0,10)
m_audio('audio_expanded_test','audio_test',cat_test,0,10)

