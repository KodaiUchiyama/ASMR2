with open('../audio/AV_model_database_test/single_TF.txt', 'r') as t:
    lines = t.readlines()
    for line in lines:
        info = line.strip().split('.')[0] # 0-0
        #num1 = info[0].strip().split('-')[1]
        #num2 = info[0].strip().split('-')[2]

        #newline = line.strip() + ' ' + num1 + '_face_emb.npy' + ' ' + num2 + '_face_emb.npy\n'
        newline = line.strip() + ' ' + info + '_face_emb.npy\n' # 0-0.npy 0-0_face_emb.npy
        with open('AVdataset_test.txt', 'a') as f:
            f.write(newline)

