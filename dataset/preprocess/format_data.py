import os
import numpy as np
import random
import pickle
from scipy.io import loadmat
from easydict import EasyDict

from sentence_transformers import SentenceTransformer
np.random.seed(0)
random.seed(0)

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
def get_label_embeds(labels):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(labels)
    return embeddings
attr_words = ['Female','Adult','LongHair','Hat','Shortsleeves','Trousers','Skirt','Short','Backpack']
print(get_label_embeds(attr_words))
def generate_data_description(save_dir, reorder, new_split_path, version):
    data = loadmat('data_annotation_shorten.mat')
    data = data['RAP_annotation']
    dataset = EasyDict()
    dataset.description = 'rap2'
    dataset.root = os.path.join(save_dir, 'data_relabel/Image')
    dataset.image_name = [data['name'][0][0][i][0][0] for i in range(len(data['name'][0][0]))]
    raw_attr_name = [data['attribute'][0][0][i][0][0] for i in range(len(data['attribute'][0][0]))]
    raw_label = data['data'][0][0]
    selected_attr_idx = (data['selected_attribute'][0][0][0] - 1).tolist() 
    print(selected_attr_idx)
    color_attr_idx = []
    # color_attr_idx = list(range(15, 41))
    #extra_attr_idx = np.setdiff1d(range(41), color_attr_idx + selected_attr_idx).tolist()
    extra_attr_idx = []
    print(color_attr_idx)
    print(extra_attr_idx)
    dataset.label = raw_label[:, selected_attr_idx + color_attr_idx + extra_attr_idx]  # (n, 119)
    print(np.array(dataset.label).shape)
    dataset.attr_name = [raw_attr_name[i] for i in selected_attr_idx + color_attr_idx + extra_attr_idx]
    #print(dataset.attr_name)
    dataset.label_idx = EasyDict()
    dataset.label_idx.eval = list(range(len(selected_attr_idx)))  # num of selected attributes
    dataset.label_idx.color = []  # not aligned with color label index in label
    dataset.label_idx.extra = []  # not aligned with extra label index in label
    dataset.attr_words = np.array(attr_words)
    dataset.attr_vectors = get_label_embeds(attr_words)
    dataset.partition = EasyDict()
    dataset.partition.train = []
    dataset.partition.val = []
    #dataset.partition.test = []
    dataset.partition.trainval = []

    dataset.weight_train = []
    dataset.weight_trainval = []

    if new_split_path:

        # remove Age46-60
        # dataset.label_idx.eval.remove(38)  # 54

        with open(new_split_path, 'rb+') as f:
            new_split = pickle.load(f)

        train = np.array(new_split.train)
        val = np.array(new_split.val)
        #test = np.array(new_split.test)
        trainval = np.concatenate((train, val), axis=0)

        print('Shape of train is: ')
        print(train.shape)
        print('Shape of val is: ')
        print(val.shape)
        print('Shape of trainval is: ')
        print(trainval.shape)


        dataset.partition.train = train
        dataset.partition.val = val
        dataset.partition.trainval = trainval
        #dataset.partition.test = test

        weight_train = np.mean(dataset.label[train], axis=0).astype(np.float32)
        weight_val = np.mean(dataset.label[val], axis=0).astype(np.float32)
        #weight_test = np.mean(dataset.label[test], axis=0).astype(np.float32)
        weight_trainval = np.mean(dataset.label[trainval], axis=0).astype(np.float32)
        print(','.join(f'{q:2f}' for q in weight_train.tolist()[:len(selected_attr_idx)]))
        print('-'*60)
        print(','.join(f'{q:2f}' for q in weight_val.tolist()[:len(selected_attr_idx)]))
        print('-'*60)
        #print(','.join(f'{q:2f}' for q in weight_test.tolist()[:len(selected_attr_idx)]))

        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)
        with open(f'data_format_v1_relabel.pkl', 'wb+') as f:
            pickle.dump(dataset, f)

    # else:
    #     for idx in range(5):
    #         train = data['partition_attribute'][0][0][0][idx]['train_index'][0][0][0] - 1
    #         val = data['partition_attribute'][0][0][0][idx]['val_index'][0][0][0] - 1
    #         test = data['partition_attribute'][0][0][0][idx]['test_index'][0][0][0] - 1
    #         trainval = np.concatenate([train, val])
    #         dataset.partition.train.append(train)
    #         dataset.partition.val.append(val)
    #         dataset.partition.test.append(test)
    #         dataset.partition.trainval.append(trainval)
    #         # cls_weight
    #         weight_train = np.mean(dataset.label[train], axis=0)
    #         weight_trainval = np.mean(dataset.label[trainval], axis=0)
    #         dataset.weight_train.append(weight_train)
    #         dataset.weight_trainval.append(weight_trainval)
    #     with open('dataset_all.pkl', 'wb+') as f:
    #         pickle.dump(dataset, f)


if __name__ == "__main__":
    reorder = False
    save_dir = '/home/cist-poc01/tuantran/VTB/data'
    new_split_path = 'data_zero_shot_split.pkl'
    generate_data_description(save_dir, reorder, new_split_path, 0)
