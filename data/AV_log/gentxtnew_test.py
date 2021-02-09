'''
with open('../audio/AV_model_database_expanded_test/single_TF.txt', 'r') as t:
    lines = t.readlines()
    for line in lines:
        info = line.strip().split('.')[0] # 0-0

        newline = line.strip() + ' ' + info + '_face_emb.npy\n' # 0-0.npy 0-0_face_emb.npy
        with open('AVdataset_expanded_test.txt', 'a') as f:
            f.write(newline)
'''
with open('../audio/AV_model_database_expanded_test/single_TF.txt', 'r') as t:
    lines = t.readlines()
    for line in lines:
        info = line.strip().split('.')[0] # 0-0
        # 交差検証を行うためtestデータの名称が，trainデータと重複するしているのを修正する
        # video index 0->9 を 35->44に変更
        num1Str = info.split('-')[0]
        num2Str = info.split('-')[1]
        #string をint変換して足す
        num1Str = str(int(num1Str) + 35)

        newline = num1Str + '-' + num2Str + '.npy' + ' ' + num1Str + '-' + num2Str + '_face_emb.npy\n' # 35-0.npy 35-0_face_emb.npy
        with open('AVdataset_expanded_test.txt', 'a') as f:
            f.write(newline)
