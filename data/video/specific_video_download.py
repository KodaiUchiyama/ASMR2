from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import subprocess
import datetime
sys.path.append("../lib")
import AVHandler as avh
import pandas as pd


def video_download(loc,cat,start_idx,end_idx):
    # Only download the video from the link
    # loc        | the location for downloaded file
    # v_name     | the name for the video file
    # cat        | the catalog with audio link and time
    # start_idx  | the starting index of the video to download
    # end_idx    | the ending index of the video to download

    for i in range(start_idx,end_idx):
        command = 'cd %s;' % loc
        f_name = str(i)
        link = avh.m_link(cat.loc[i, 'link'])
        start_time = cat.loc[i, 'start_time']
        end_time = start_time + 3.0
        start_time = datetime.timedelta(seconds=start_time)
        end_time = datetime.timedelta(seconds=end_time)
        command += 'ffmpeg -i $(youtube-dl -f ”mp4“ --get-url ' + link + ') ' + '-c:v h264 -c:a copy -ss %s -to %s %s.mp4' \
                % (start_time, end_time, f_name)
        #command += 'ffmpeg -i %s.mp4 -r 25 %s.mp4;' % (f_name,'clip_' + f_name) #convert fps to 25
        #command += 'rm %s.mp4' % f_name
        os.system(command)

def generate_frames(loc,start_idx,end_idx):
    # get frames for each video clip
    # loc        | the location of video clip
    # v_name     | v_name = 'clip_video_train'
    # start_idx  | the starting index of the training sample
    # end_idx    | the ending index of the training sample

    avh.mkdir('frames')
    for i in range(start_idx, end_idx):
        command = 'cd %s;' % loc
        f_name = str(i)
        command += 'ffmpeg -i %s.mp4 -y -f image2  -vframes 75 ../frames/%s-%%02d.jpg' % (f_name, f_name)
        os.system(command)


def download_video_frames(loc,cat,output_file,start_idx,end_idx,rm_video):
    # Download each video and convert to frames immediately, can choose to remove video file
    # loc        | the location for downloaded file
    # cat        | the catalog with audio link and time
    # start_idx  | the starting index of the video to download
    # end_idx    | the ending index of the video to download
    # rm_video   | boolean value for delete video and only keep the frames

    #avh.mkdir(output_file)
    for i in range(start_idx, end_idx):
        command = 'cd %s;' % loc
        f_name = str(i)
        link = avh.m_link(cat.loc[i, 'link'])
        start_time = cat.loc[i, 'start_time']
        end_time = cat.loc[i,'end_time']
        start_time_sec = int(start_time.strip().split("-")[0])*60+int(start_time.strip().split("-")[1])
        end_time_sec = int(end_time.strip().split("-")[0])*60+int(end_time.strip().split("-")[1])
        
        print('index_num:%s'%i)
        win_num =int((end_time_sec - start_time_sec) / 3)
        print('segment_size:%s' % win_num)
         
        command += 'ffmpeg -i $(youtube-dl -f ”mp4“ --get-url ' + link + ') ' + '-vcodec libx264 -acodec copy %s.mp4;' %f_name
        #try:
        #    output = subprocess.check_output(command)
        #except subprocess.CalledProcessError as e:
        #    print(f"returncode:{e.returncode}, output:{e.output}")
        
        os.system(command)
        command2 = 'cd %s;' % loc
        for j in range(win_num):
            segment_video_name = str(i) + "-" + str(j)
            #start_time = datetime.timedelta(seconds=start_time_sec)
            #end_time = datetime.timedelta(seconds=start_time_sec+3)
            #command += 'ffmpeg -i $(youtube-dl -f ”mp4“ --get-url ' + link + ') ' + '-vcodec libx264 -codec:a copy -ss %s -to %s %s.mp4 < /dev/null;' % (start_time, end_time, f_name)
            #split video to 3 sec segment
            command2 += 'ffmpeg -ss %s -i %s.mp4 -t 3 -vcodec libx264 -codec:a copy %s.mp4;' %(start_time_sec, f_name, segment_video_name)
            
            #test用に25fpsの動画を作成する
            command2 += 'ffmpeg -i %s.mp4 -vf fps=25 -vcodec libx264 -acodec copy fps25_%s.mp4;' % (segment_video_name, segment_video_name)
            
            #converts to frames
            #「-vf」(video filter)オプション
            #command2 += 'ffmpeg -i %s.mp4 -vf fps=25 ../%s/%s-%%02d.jpg;' % (segment_video_name,output_file, segment_video_name)
            
            #increment start_time_sec
            start_time_sec = start_time_sec + 3
        
        os.system(command2)
        if rm_video:
            # delete each segment video
            os.system("rm ./%s/%s-*.mp4"%(loc,f_name))
            # delete original video
            os.system("rm ./%s/%s.mp4"%(loc,f_name))

loc_file_name='video_train_separation'
avh.mkdir(loc_file_name)
cat_train = pd.read_csv('../audio/catalog/avspeech_train.csv')
#cat_test = pd.read_csv('../audio/catalog/avspeech_test.csv')
output_file = 'frames_test'

# download video , convert to images separately
#video_download(loc=loc_file_name,v_name='video_train',cat=cat_train,start_idx=2,end_idx=4)
#avh.generate_frames(loc='video_train',v_name='clip_video_train',start_idx=2,end_idx=4)

# download each video and convert to frames immediately
#download_video_frames(loc='video_train',cat=cat_train,start_idx=33,end_idx=35,rm_video=True)
download_video_frames(loc=loc_file_name,cat=cat_train, output_file=output_file, start_idx=26,end_idx=27,rm_video=False)
