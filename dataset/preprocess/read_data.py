
from pandas import read_excel
import numpy as np
import os
import shutil
import scipy.io
from tqdm import tqdm
rap2_dict = {'RAP_annotation': {}}
data_dict = {'name': [], 'data': [],
            'attribute': [['Female'],['Child'],['Adult'],['Old'],['LongHair'],['Glasses'],['Hat'],
                        ['Shortsleeves'],['Trousers'],['Jean'],['Skirt'],['Short'],['Backpack'],
                        ['Bag'],['attOther'],['ub_black'],['ub_white'],['ub_gray'],
                        ['ub_red'],['ub_green'],['ub_blue'],['ub_yellow'],['ub_brown'],
                        ['ub_purple'],['ub_pink'],['ub_orange'],['ub_mix'],['ub_other'],
                        ['lb_black'],['lb_white'],['lb_gray'],['lb_red'],['lb_green'],
                        ['lb_blue'],['lb_yellow'],['lb_brown'],['lb_purple'],['lb_pink'],
                        ['lb_orange'],['lb_mix'],['lb_other']],
            'selected_attribute': [1,3,5,7,8,9,11,12,13]}

data_dict['attribute'] = np.asarray(data_dict['attribute'], dtype='object')
#print(data_dict['attribute'].shape)
data_all = []
count = 0
for label_name in tqdm(os.listdir('./data_re_label/Label')):
    name = []
    pid = label_name.split('_')[0]
    if int(pid)>=3652:
        count += 1
        img_name = label_name
        img_name = img_name.replace('.txt', '.jpg')
        #print(img_name)
    else:
        img_name = label_name.replace(pid+'_','')
        if 'CAM' in img_name:
            img_name = img_name.replace('.txt', '.png')
        else:
            img_name = img_name.replace('.txt', '.jpg')
    # if ".txt.jpg" in img_name:
    #     img_name = img_name.replace('.txt', '.jpg')
    name.append(img_name)
    f = open('./data_re_label/Label/'+label_name,'r')
    content = f.read().split(',')
    f.close()
    # permutation in raw label file here for collapse for combine some attributes together
    file_data = [1 if elem =='1' else 0 for elem in content]
    
    if (file_data[3] == 1): file_data[2] = 1            # combine Old into Adult
    if (file_data[9] == 1): file_data[8] = 1            # combine Jean into Trousers
    if (file_data[13] == 1): file_data[14] = 1          # combine Backpack into  Att-other
    #if (file_data[13] == 1): file_data[12] = 1          # combine Bag with Backpack
    
    #if (file_data[22] == 1 or file_data[25] == 1): file_data[21] = 1        # Yellow, Brown, Orange
    #if (file_data[23] == 1): file_data[20] = 1          # purple + blue
    #if (file_data[24] == 1): file_data[18] = 1          # red + pink

    #if (file_data[35] == 1 or file_data[38] == 1): file_data[34] = 1        # Yellow, Brown, Orange
    #if (file_data[36] == 1): file_data[33] = 1          # purple + blue
    #if (file_data[37] == 1): file_data[31] = 1          # red + pink
    data_att = []
    for i in range(len(file_data)):
        data_att.append(file_data[i])
    data_all.append(data_att)  
    #print(name)  
    data_dict['name'].append(name)
    data_dict['data'].append(file_data)
data_dict['name'] = np.asarray(data_dict['name'], dtype='object')
print(np.sum(np.array(data_all),axis = 0))
print(count)
rap2_dict['RAP_annotation'] = data_dict
#print(data_dict['name'])
scipy.io.savemat('data_annotation_shorten.mat', rap2_dict)
