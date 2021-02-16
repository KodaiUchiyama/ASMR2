import os, glob

fold_num = 'fold1'
dir_path = './result/' + fold_num +'/Marged_output_video_griffin/'
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

wav_dir = './result/'+ fold_num +'/Retrieved_wav_griffin/'
video_dir = '../../data/video/video_expanded_test/'
print('1,wav files directory : %s'%wav_dir)
print('2,video files directory : %s'%video_dir)
#command = ''
counter = 0
for wav_file_path in glob.glob(wav_dir+'*.wav'):
    #print(wav_file_path)
    segment_idx = wav_file_path.rsplit('/',1)[1].replace('.wav','') #index 35->44 ex 35-10
    # indexを揃える
    num1 = segment_idx.split('-')[0]
    num2 = segment_idx.split('-')[1]
    # indexを35->44から0-9に変更
    num1 = str(int(num1) - 35)
    video_file_path = video_dir + 'fps25_' + num1 + '-' + num2 + '.mp4' 
    print('processing index num : %s'%segment_idx)
    output_path = dir_path + 'MargedVideo_' +segment_idx + '.mp4' 
    command = 'ffmpeg -i %s -i %s -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 %s' %(video_file_path, wav_file_path, output_path)
    os.system(command)
    
    counter += 1
    if counter == 100:
        print('process counter:'+str(counter))
#print(command)
#os.system(command)


