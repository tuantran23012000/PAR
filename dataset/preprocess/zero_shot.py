import numpy as np 
import os
import shutil
from scipy.io import loadmat
import pickle
from easydict import EasyDict
from tqdm import tqdm
label_file = []
I = []
count = 0
for name in os.listdir('./data_re_label/Label'):
    label_file.append(name)
    pid = int(name.split('_')[0])
    count += 1
    if pid not in I:
        I.append(pid)

K = max(I)
print(K)
Tid = 50
Tattr = 0.03
Timg = 300
Id2Idx = {}
img_threshold = (3/5)*count #55202
# name of matlab file contain all infomation of dataset
annotation = loadmat('data_annotation_shorten_61689.mat', )
img_array = annotation['RAP_annotation']['name']
selected_attr_idx = annotation['RAP_annotation']['selected_attribute'][0][0][0]
img_list = []

for i in range(len(img_array[0][0])):
    img_list.append(img_array[0][0][i][0][0])


for k in tqdm(I):
    idx = {}
    idx['idx'] = []
    idx['img'] = []
    idx['label'] = []
    for name in label_file:
        if (int(name.split('_')[0]) == k):
            pid = name.split('_')[0]
            if int(pid) >= 3652:
                
                img_name = name
                img_name = img_name.replace('.txt','.jpg')
                
            else:
                img_name = name.replace(pid+'_','')
                if 'CAM' in img_name:
                    img_name = img_name.replace('txt','png')
                else:
                    img_name = img_name.replace('txt','jpg')
            idx['img'].append(img_name)
            idx['idx'].append(img_list.index(img_name))
            f = open('./data_re_label/Label/'+name, 'r')
            label = f.read().split(',')
            idx['label'].append([1 if elem == '1' else 0 for elem in label])
            f.close()
    Id2Idx[k] = idx


split_time = 1
while(1):
    shrink_I = I.copy()
    #print('Perform zero-shot split no. ' + str(split_time))
    K_train = np.random.randint(int(4*K/5-Tid), int(4*K/5+Tid))
    #print(K_train)
    #K_train = 24252
    I_train = np.random.choice(shrink_I, K_train, replace = False)
    K_mid = int((K-K_train)/2)
    #K_valid = np.random.randint(K_mid-Tid, K_mid+Tid)
    K_valid = K - K_train
    for elem in I_train:
        shrink_I.remove(elem)
    I_valid = shrink_I
    # I_valid = np.random.choice(shrink_I, K_valid, replace = False)
    # K_test = K - K_train - K_valid
    # for elem in I_valid:
    #     shrink_I.remove(elem)
    #I_test = shrink_I
    D_train = {}
    D_valid = {}
    #D_test = {}
    num_valid_img = 0
    num_train_img = 0
    for k in I:
        if k in I_train:
            D_train[k] = Id2Idx[k]
            num_train_img += len(Id2Idx[k]['img'])
        if k in I_valid:
            D_valid[k] = Id2Idx[k]
            num_valid_img += len(Id2Idx[k]['img'])
        # if k in I_test:
        #     D_test[k] = Id2Idx[k]
        #     num_test_img += len(Id2Idx[k]['img'])
    print(num_train_img , num_valid_img)
    if abs(num_train_img - num_valid_img) < img_threshold:
        split_time += 1
        print('Not compatible size of test-valid')
        continue
    y_train = []
    y_valid = []
    #y_test = []
    for elem in D_train.keys():
        for label in D_train[elem]['label']:
            y_train.append(label)
    for elem in D_valid.keys():
        for label in D_valid[elem]['label']:
            y_valid.append(label)
    # for elem in D_test.keys():
    #     for label in D_test[elem]['label']:
    #         y_test.append(label)

    train_ratio = np.mean(y_train,axis = 0)
    valid_ratio = np.mean(y_valid,axis = 0)
    # test_ratio = np.mean(y_test,axis = 0, dtype = object)

    train_val = [(abs(train_ratio[i-1]-valid_ratio[i-1]) < Tattr) for i in selected_attr_idx]
    #train_test = [(abs(train_ratio[i-1]-test_ratio[i-1]) < Tattr) for i in selected_attr_idx]
    #if (all(train_val) and all(train_test)):
    if all(train_val):
        print('Found a solution')
        break
    print('Positive ratio not sufficient')
    split_time += 1

print('Train size: ' + str(len(y_train)))
print('Val size: ' + str(len(y_valid)))
#print('Test size: ' + str(len(y_test)))
# print(train_ratio[:15])
# print(valid_ratio[:15])
# print(test_ratio[:15])

train = []
val = []
#test = []
for elem in D_train.keys():
    for idx in D_train[elem]['idx']:
        train.append(idx)
for elem in D_valid.keys():
    for idx in D_valid[elem]['idx']:
        val.append(idx)
# for elem in D_test.keys():
#     for idx in D_test[elem]['idx']:
#         test.append(idx)

pickle_data = EasyDict()
pickle_data.train = train
pickle_data.val = val
#pickle_data.test = test
pickle.dump(pickle_data, open('data_zero_shot_split.pkl', 'wb'))
