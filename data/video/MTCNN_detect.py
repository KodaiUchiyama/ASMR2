from mtcnn.mtcnn import MTCNN
import cv2
import pandas as pd
import os
import glob

frame_path = './frames_expanded_test/'
output_dir = './face_input_expanded_test'
#invalid_frame_path = 'invalid_frame2.txt'
segment_num_list = 'segment_num_list_expanded_test.csv'
detect_range = (0,10)

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

def bounding_box_check(faces):
    # check the center
    for face in faces:
        bounding_box = face['box'] #The bounding box is formatted as [x, y, width, height] under the key 'box'.
        if(bounding_box[1]<0):
            bounding_box[1] = 0
        if(bounding_box[0]<0):
            bounding_box[0] = 0
        #if(bounding_box[0]-50>x or bounding_box[0]+bounding_box[2]+50<x):
        #    print('change person from')
        #    print(bounding_box)
        #    print('to')
        #    continue
        #if (bounding_box[1]-50 > y or bounding_box[1] + bounding_box[3]+50 < y):
        #    print('change person from')
        #    print(bounding_box)
        #    print('to')
        #    continue
        return bounding_box

def face_detect(file,detector,frame_path=frame_path):
    name = file.replace('.jpg', '').rsplit('-',1)
    #x = log['pos_x'] #Where the X,Y coordinates mark the center point of the speaker's face in the frame at the beginning of the segment(3s video)
    #y = log['pos_y']
    
    # 通常フレームload
    img = cv2.imread('%s%s'%(frame_path,file))
    #x = img.shape[1] * x #the X,Y coordinates are normalised
    #y = img.shape[0] * y
    faces = detector.detect_faces(img)
    # check if detected faces
    if(len(faces)==0):
        print('no face detect: '+file)
        #顔が認識できなかったファイルをログに出力
        #with open(invalid_frame_path,'a') as f:
        #    f.write(file+'\n')
        #return #no face
    #bounding_box = bounding_box_check(faces,x,y)
    bounding_box = bounding_box_check(faces)
    if(bounding_box == None):
        print('face is not related to given coord: '+file)
        return
    print(file," ",bounding_box)
    #print(file," ",x, y)
    crop_img = img[bounding_box[1]:bounding_box[1] + bounding_box[3],bounding_box[0]:bounding_box[0]+bounding_box[2]]
    crop_img = cv2.resize(crop_img,(160,160))
    cv2.imwrite('%s/frame_'%output_dir + name[0] + '_' + name[1] + '.jpg', crop_img) #face_input/frame_0-14_15.jpg
    #crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    #plt.imshow(crop_img)
    #plt.show()


detector = MTCNN()
for i in range(detect_range[0],detect_range[1]):
    # "{index}-"で始まるファイルのみを取得
    counter = 0
    #segment_video_frame_list = glob.glob('frames/%s-%s-*'%(i,counter))
    while len(glob.glob('%s%s-%s-*'%(frame_path,i,counter))) > 0:
        #0-1,0-2,,のセグメントの数ループ
        for j in range(1,76):
            file_name = "%s-%s-%02d.jpg"%(i ,counter ,j)#ex)0-0-57.jpg
            # 25fps分割された各フレームが存在するかチェック
            if (not os.path.exists('%s%s' % (frame_path, file_name))):
                print('cannot find input: ' + '%s%s' % (frame_path, file_name))
                continue
            face_detect(file_name, detector)
        #increment counter
        print("COUNTER : %s"%counter)
        counter+=1
    #各動画に3sのセグメントがいくつあるかを記録
    with open(segment_num_list,'a') as l:
        l.write('%d,%d\n'%(i,counter-1)) #NOTE:counterが最後に1インクリメントするから -1する 


