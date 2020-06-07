import os, glob
#import pandas as pd
#inspect_dir = 'face_input'
#inspect_range = (0,17)
#valid_frame_path = 'valid_frame.txt'
#AfewMissed_frame_path = 'AfewMissd_frame.txt'
#segment_num_list = pd.read_csv('segment_num_list.csv')

dir_path = './Marged_output_video/'
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

wav_dir = './Retrieved_predicted_wav/'
video_dir = '../../data/video/video_test/'
print('1,wav files directory : %s'%wav_dir)
print('2,video files directory : %s'%video_dir)
command = ''
for wav_file_path in glob.glob(wav_dir+'*.wav'):
    #print(wav_file_path)
    segment_idx = wav_file_path.rsplit('/',1)[1].replace('.wav','')
    video_file_path = video_dir + 'fps25_' + segment_idx + '.mp4' 
    print('processing index num : %s'%segment_idx)
    output_path = dir_path + 'MargedVideo_' +segment_idx + '.mp4' 
    command += 'ffmpeg -i %s -i %s -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 %s;' %(video_file_path, wav_file_path, output_path)

#print(command)
os.system(command)


