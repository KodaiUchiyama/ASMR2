import os, glob
import pandas as pd
inspect_dir = 'face_input_expanded_test'
inspect_range = (0,10)
valid_frame_path = 'valid_frame_expanded_test.txt'
#AfewMissed_frame_path = 'AfewMissd_frame.txt'
segment_num_list = pd.read_csv('segment_num_list_expanded_test.csv')

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
    
    print('processing: video %s'%i)
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
                    #invalidなフレームを削除
                    #path = inspect_dir + "/frame_%d_*.jpg"% i
                    #for file in glob.glob(path):
                    #    os.remove(file)
                    valid = False
                    invalid_frame_counter+=1
                    #print('frame %s is not valid'%i)
                    #break
            #print("video %s, segment %s has %s invalid frames"%(i,counter,invalid_frame_counter))
            #if(invalid_frame_counter == 2 or invalid_frame_counter == 1):
            #    with open(AfewMissed_frame_path,'a') as f:
            #        line = "video %s, segment %s has %s invalid frames"%(i,counter,invalid_frame_counter)
            #        f.write(line+'\n')
            if valid:
                with open(valid_frame_path,'a') as f:
                    frame_name = "frame_%d-%d"%(i,counter)
                    f.write(frame_name+'\n')
        #increment segment counter
        counter+=1


