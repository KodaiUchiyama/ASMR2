#frame_interpolate_checker.pyより得た、1frameのみ欠損しているセグメントを補間する
import os, glob
import pandas as pd
import cv2

inspect_dir = './face_input'
output_dir = './output_interpolation'
inspect_range = (0,35)
#valid_frame_path = 'valid_frame.txt'
AfewMissed_frame_path = 'AfewMissd_frame.txt'
segment_num_list = pd.read_csv('segment_num_list.csv')

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

#face画像はあるかどうかをbooleanで返す
def check_frame(idx,segment,frame,dir=inspect_dir):
    path = dir + "/frame_%s-%s_%02d.jpg"%(idx,segment,frame)
    if(not os.path.exists(path)): return False
    return True

def interpolate_frame(video_id, segment_id, frame_id, dir=inspect_dir, output_dir=output_dir):
    imgA_path = dir + "/frame_%s-%s_%02d.jpg"%(video_id,segment_id,frame_id-1)
    imgC_path = dir + "/frame_%s-%s_%02d.jpg"%(video_id,segment_id,frame_id+1)
    imgA = cv2.imread(imgA_path)
    imgC = cv2.imread(imgC_path)
    #print(imgA.dtype)
    #print(imgC.dtype)
    #print(imgA.shape)
    imgB = imgA/2.0 + imgC/2.0 # generate pseudo frame 
    cv2.imwrite(output_dir+"/frame_%s-%s_%02d.jpg"%(video_id,segment_id,frame_id), imgB)  # Save a frame

with open(AfewMissed_frame_path, 'r') as t:
    lines = t.readlines()
    for line in lines:
        video_id = line.strip().split(',')[0] 
        segment_id = line.strip().split(',')[1]
        #print(glob.glob(inspect_dir+'/frame_%s-%s_*'%(video_id, segment_id)))
        #for path in glob.glob(inspect_dir+'/frame_%s-%s_*'%(video_id, segment_id)):
        for j in range(1,76):
            if(check_frame(video_id,segment_id,j)==False):
                if(j == 1 or j ==75):
                    continue
                interpolate_frame(video_id, segment_id, j)
                print('interpolating: video:%s,segment:%s,frame:%s'%(video_id,segment_id,j))
                #valid = False
                #invalid_frame_counter+=1
                #print('frame %s is not valid'%i)
                #break    


exit()
#face画像はあるかどうかをbooleanで返す
def check_frame(idx,segment,frame,dir=inspect_dir):
    path = dir + "/frame_%s-%s_%02d.jpg"%(idx,segment,frame)
    if(not os.path.exists(path)): return False
    return True

for i in range(inspect_range[0],inspect_range[1]):
    counter = 0
    #the number of segments which each video has
    #add "id,segment" at the head of segment_num_list_csv
    segment_num = segment_num_list.loc[i,'segment']
    
    print('processing video %s'%i)
    #if(len(glob.glob(inspect_dir+'/frame_%s-%s_*'%(i,counter))) > 0):
    while counter <= segment_num:
        #no segment at the video
        if(len(glob.glob(inspect_dir+'/frame_%s-%s_*'%(i,counter))) == 0):
            print("video %s, segment %s has no face frames"%(i,counter)) 
        else:
            valid = True
            invalid_frame_counter = 0
            #print('processing video %s, segment %s'%(i,counter))
            for j in range(1,76):
                if(check_frame(i,counter,j)==False):
                    
                    valid = False
                    invalid_frame_counter+=1
                    #print('frame %s is not valid'%i)
                    #break
            #print("video %s, segment %s has %s invalid frames"%(i,counter,invalid_frame_counter))
            if(invalid_frame_counter == 1):
                with open(AfewMissed_frame_path,'a') as f:
                    line = "video %s, segment %s has %s invalid frames"%(i,counter,invalid_frame_counter)
                    f.write(line+'\n')
            #if valid:
            #    with open(valid_frame_path,'a') as f:
            #        frame_name = "frame_%d-%d"%(i,counter)
            #        f.write(frame_name+'\n')
        #increment segment counter
        counter+=1


