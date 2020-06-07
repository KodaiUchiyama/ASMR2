#frame_interpolate_checker.pyより得た、1frameのみ欠損しているセグメントを補間する
import os, glob
import pandas as pd
import cv2

inspect_dir = './face_input_test'
output_dir = './output_interpolation'
inspect_range = (0,2)
#valid_frame_path = 'valid_frame.txt'
AfewMissed_frame_path = 'AfewMissd_frame_test.txt'
segment_num_list = pd.read_csv('segment_num_list_test.csv')

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



