# import argparse
# import json
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import pickle

# from dataset.augmentation import get_transform
# from dataset.multi_label.coco import COCO14
# from metrics.pedestrian_metrics import get_pedestrian_metrics
# from models.model_factory import build_backbone, build_classifier

# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# from configs import cfg, update_config
# from dataset.pedes_attr.pedes import PedesAttr
# from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
# from models.base_block import FeatClassifier
# # from models.model_factory import model_dict, classifier_dict

# from tools.function import get_model_log_path, get_reload_weight
# from tools.utils import set_seed, str2bool, time_str
# from models.backbone import swin_transformer, resnet, bninception
# from models.backbone.tresnet import tresnet
# from losses import bceloss, scaledbceloss
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# set_seed(605)

# img_dir = './Test/Zalo/'
# img_path = './CMC/20210712_075000_000_0800_7782.jpg'
# result_dir = './Test/Zalo_Test_res/'

# def get_model(cfg, args):
#     exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
#     model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)

#     train_tsfm, valid_tsfm = get_transform(cfg)
#     #print(valid_tsfm)

#     if cfg.DATASET.TYPE == 'multi_label':
#         train_set = COCO14(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=train_tsfm,
#                            target_transform=cfg.DATASET.TARGETTRANSFORM)

#         valid_set = COCO14(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
#                            target_transform=cfg.DATASET.TARGETTRANSFORM)
#     else:
#         train_set = PedesAttr(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=valid_tsfm,
#                               target_transform=cfg.DATASET.TARGETTRANSFORM)
#         valid_set = PedesAttr(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
#                               target_transform=cfg.DATASET.TARGETTRANSFORM)

#     train_loader = DataLoader(
#         dataset=train_set,
#         batch_size=cfg.TRAIN.BATCH_SIZE,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True,
#     )

#     valid_loader = DataLoader(
#         dataset=valid_set,
#         batch_size=cfg.TRAIN.BATCH_SIZE,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True,
#     )
#     attr_name = valid_set.attr_id
#     print(attr_name)
#     backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)

#     classifier = build_classifier(cfg.CLASSIFIER.NAME)(
#         nattr=train_set.attr_num,
#         c_in=c_output,
#         bn=cfg.CLASSIFIER.BN,
#         pool=cfg.CLASSIFIER.POOLING,
#         scale =cfg.CLASSIFIER.SCALE
#     )

#     model = FeatClassifier(backbone, classifier)

#     if torch.cuda.is_available():
#         model = torch.nn.DataParallel(model).cuda()

#     model = get_reload_weight(model_dir, model, pth='ckpt_max_2022-05-15_18:55:52.pth')

#     model.eval()
#     return (model, valid_tsfm, attr_name)

# def argument_parser():
#     parser = argparse.ArgumentParser(description="attribute recognition",
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#     parser.add_argument(
#         "--cfg", help="decide which cfg to use", type=str, default="./configs/pedes_baseline/rap_zs_addition.yaml"
#     )
#     parser.add_argument("--debug", type=str2bool, default="true")

#     args = parser.parse_args()

#     return args

# args = argument_parser()
# update_config(cfg, args)

# (model, valid_tsfm, attr_name) = get_model(cfg, args)

# pred_probs = []
# img_list = []
# preds_label = []
# with torch.no_grad():
#     for img in os.listdir(img_dir):
#         img_list.append(img)
#         img_path = os.path.join(img_dir, img)
#         image = Image.open(img_path)
#         image = valid_tsfm(image).float()
#         image = image.unsqueeze(0)
#         image = image.cuda()
#         valid_logits, attns = model(image)
#         valid_probs = torch.sigmoid(valid_logits[0])
#         pred_probs.append(valid_probs.cpu().numpy())

# pred_probs = np.concatenate(pred_probs, axis=0)

# for i in range(len(pred_probs)):
#     # src_dir = os.path.join(img_dir,img_path[i])
#     # shutil.copy(src_dir, './test_img')
#     attribute = []
#     true_label = []
#     other = [4,5,11,12,13]
#     lower = False
#     #predict label
#     if pred_probs[i][0] >= 0.5: attribute.append('Female')
#     else: attribute.append('Male')
#     if pred_probs[i][1] >= 0.5: attribute.append('Child')
#     else: attribute.append('Adult')
#     if pred_probs[i][3] >= 0.5: attribute.append('Long Hair')
#     else: attribute.append('Short Hair')
#     if pred_probs[i][6] >= 0.5: attribute.append('Short sleeves')
#     else: attribute.append('Long sleeves')
#     for j in range(7,11):
#         if pred_probs[i][j] >= 0.5:
#             attribute.append(attr_name[j])
#             lower = True
#     if not lower:
#         attribute.append(attr_name[pred_probs[i].tolist().index(max(pred_probs[i][7:11]))])
#     for j in other:
#         if pred_probs[i][j] >= 0.5:
#             attribute.append(attr_name[j])
#     img = mpimg.imread(img_dir+img_list[i])
#     imgplot = plt.imshow(img)
#     plt.title(attribute)
#     x_axis = imgplot.axes.get_xaxis()
#     y_axis = imgplot.axes.get_yaxis()
#     x_axis.set_visible(False)
#     y_axis.set_visible(False)
#     plt.savefig(result_dir+img_list[i])
#     # plt.show()
#     plt.close()
import argparse
import json
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle

from dataset.augmentation import get_transform
from dataset.multi_label.coco import COCO14
from metrics.pedestrian_metrics import get_pedestrian_metrics
from models.model_factory import build_backbone, build_classifier

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import cfg, update_config
from dataset.pedes_attr.pedes import PedesAttr
from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
from models.base_block import FeatClassifier
# from models.model_factory import model_dict, classifier_dict

from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool, time_str
from models.backbone import swin_transformer, resnet, bninception
#from models.backbone.tresnet import tresnet
from losses import bceloss, scaledbceloss
from PIL import Image
set_seed(605)

# img_name = '18977.png'
# result_dir = '/home/tuantran/huyeniot/Projects_attribute/Rethinking_of_PAR/data/PETA/'
cfg_file = "./configs/pedes_baseline/rap_zs_addition.yaml"

def get_model(cfg, args):
    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)

    train_tsfm, valid_tsfm = get_transform(cfg)
    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)

    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=10,
        c_in=c_output,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale =cfg.CLASSIFIER.SCALE
    )

    model = FeatClassifier(backbone, classifier)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    model = get_reload_weight(model_dir, model, pth= 'ckpt_max_2022-09-13_17:18:32.pth')#'ckpt_max_2022-09-07_17:31:30.pth') #'ckpt_max_2022-08-25_15:34:55.pth')

    model.eval()
    return model, valid_tsfm, model_dir

def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str, default=cfg_file
    )
    parser.add_argument("--debug", type=str2bool, default="true")

    args = parser.parse_args()

    return args

args = argument_parser()
update_config(cfg, args)

model, valid_tsfm,model_dir = get_model(cfg, args)

# pred_probs = []
img_list = []
preds_label = []
# ['Female'],['Child'],['Adult'],['Old'],['LongHair'],['Glasses'],['Hat'],
#                         ['Shortsleeves'],['Trousers'],['Jean'],['Skirt'],['Short'],['Backpack'],
#                         ['Bag'],['attOther'] // 0,2,4,6,7,8,10,11,12,14

# ['Female'],['Adult'],['LongHair'],['Hat'],['Shortsleeves'],['Trousers-Jean'],['Skirt'],['Short'],['Backpack'],
#                         ['Bag-attOther']
a = [2,3,4,6,7,8,9]
count = 0
img_dir = '/media/cist/NAS/bk_thanh_huyen/huyeniot/Projects_attribute/Rethinking_of_PAR/data/CMC/Image'
label_dir = '/media/cist/NAS/bk_thanh_huyen/huyeniot/Projects_attribute/Rethinking_of_PAR/data/CMC/Label'
pred_probs = []
preds_label = []
gt_label = []
count = 0
gt_label = []
path_list = []
# for label_name in os.listdir(label_dir):
#     f = open(os.path.join(label_dir,label), 'r')
#     label = f.read().split(',')
#     gt_label
with torch.no_grad():
    for img in tqdm(os.listdir(img_dir)):
        path_list.append(img)
        img_path = os.path.join(img_dir, img)
        image = Image.open(img_path)
        image = valid_tsfm(image).float()
        image = image.unsqueeze(0)
        image = image.cuda()
        valid_logits, attns = model(image)
        valid_probs = torch.sigmoid(valid_logits[0])
        pred_probs.append(valid_probs.cpu().numpy()[:,0:9])
        #print(valid_probs.cpu().numpy().shape)
        #print(valid_probs.cpu().numpy()[:,0:8].shape)
        #print(valid_probs.cpu().numpy())
        f = open(os.path.join(label_dir,img.replace('.jpg','.txt')), 'r')
        label = f.read().split(',')
        file_data = [1 if elem =='1' else 0 for elem in label]
        if (file_data[3] == 1): file_data[2] = 1            # combine Old into Adult
        if (file_data[9] == 1): file_data[8] = 1            # combine Jean into Trousers
        if (file_data[13] == 1): file_data[14] = 1          # combine Backpack into  Att-other
        data_att = []
        for i in range(len(file_data)):
            if i != 1 and i != 3 and i!=5 and i!=9 and i!=13 and i<=12:
                data_att.append(file_data[i])
        #print(np.array(data_att).shape)
        #data_all.append(data_att)  
        #gt_label.append(np.array([[1 if elem == '1' else 0 for elem in label]]))
        #gt_label.append(np.array([[label[0],label[2],label[0]]))
        gt_label.append(np.array([data_att]))
        count += 1
    # for img in tqdm(os.listdir(img_dir)):
    #     path_list.append(img)
    #     img_path = os.path.join(img_dir, img)
    #     image = Image.open(img_path)
    #     image = valid_tsfm1(image).float()
    #     image = image.unsqueeze(0)
    #     image = image.cuda()
    #     valid_logits, attns = model(image)
    #     valid_probs = torch.sigmoid(valid_logits[0])
    #     pred_probs.append(valid_probs.cpu().numpy())
    #     #print(valid_probs.cpu().numpy())
    #     f = open(os.path.join(label_dir,img.replace('.jpg','.txt')), 'r')
    #     label = f.read().split(',')
    #     file_data = [1 if elem =='1' else 0 for elem in label]
    #     if (file_data[3] == 1): file_data[2] = 1            # combine Old into Adult
    #     if (file_data[9] == 1): file_data[8] = 1            # combine Jean into Trousers
    #     if (file_data[13] == 1): file_data[14] = 1          # combine Backpack into  Att-other
    #     data_att = []
    #     for i in range(len(file_data)):
    #         if i != 1 and i != 3 and i!=5 and i!=9 and i!=13 and i<=14:
    #             data_att.append(file_data[i])
    #     #data_all.append(data_att)  
    #     #gt_label.append(np.array([[1 if elem == '1' else 0 for elem in label]]))
    #     #gt_label.append(np.array([[label[0],label[2],label[0]]))
    #     gt_label.append(np.array([data_att]))
    #     count += 1
    #     # if count == 2:
    #     #     break
preds_probs = np.concatenate(pred_probs, axis=0)
gt_label = np.concatenate(gt_label, axis=0)
# print(pred_probs)
# print(gt_label)

threshold = 0.5
preds_label = []
for i in range(len(preds_probs)):
    predicted = [1 if (attribute >= threshold) else 0 for attribute in preds_probs[i]]
    preds_label.append(predicted)
valid_result = get_pedestrian_metrics(gt_label, preds_probs,threshold=threshold)
valid_map, _ = get_map_metrics(gt_label, preds_probs)

print(f'Evaluation on test set, \n',
    'ma: {:.4f},  map: {:.4f}, label_f1: {:4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
        valid_result.ma, valid_map, np.mean(valid_result.label_f1), np.mean(valid_result.label_pos_recall),
        np.mean(valid_result.label_neg_recall)),
    'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
        valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
        valid_result.instance_f1)
    )


TP = [0]*len(preds_label[0])
TN = [0]*len(preds_label[0])
FP = [0]*len(preds_label[0])
FN = [0]*len(preds_label[0])
positive = [0]*len(preds_label[0])
negative = [0]*len(preds_label[0])

for i in range(len(preds_label)):
    for j in range(len(preds_label[i])):
        if (gt_label[i][j]==1):
            positive[j] += 1
            if (preds_label[i][j] == gt_label[i][j]): 
                TP[j] += 1
            else: 
                FN[j] += 1
        if (gt_label[i][j]==0):
            negative[j] += 1
            if (preds_label[i][j] == gt_label[i][j]):
                TN[j] += 1
            else:
                FP[j] += 1

for i in range(len(TP)):
    # if (positive[i] == 0) or (negative[i] == 0):
    #     continue
    # TP[i] /= positive[i]
    # #FP[i] /= negative[i]
    # #TN[i] /= negative[i]
    # FN[i] /= positive[i]
    if positive[i] != 0:
        TP[i] /= positive[i]
        FN[i] /= positive[i]
    if negative[i] != 0:
        FP[i] /= negative[i]
        TN[i] /= negative[i]
    if (positive[i] == 0):
        TP[i] = 0 
        #FP[i] = 0 
        #TN[i] = 0
        FN[i] = 0 
        #continue
    if  (negative[i] == 0):
        FP[i] = 0 
        TN[i] = 0
print('Here is the result: ')
print('-'*60)
print('True positive:')
print(', '.join(f'{q:.2f}' for q in TP))
print('-'*60)
print('False positive:')
print(', '.join(f'{q:.2f}' for q in FP))
print('-'*60)
print('True negative:')
print(', '.join(f'{q:.2f}' for q in TN))
print('-'*60)
print('False negative:')
print(', '.join(f'{q:.2f}' for q in FN))
# attn_list = []
# t_list =  []
# MA_list = []
# ACC_list = []
# PRE_list = []
# REC_list = []
# F1_list = []
# for t in range(50,100,5):
#     threshold = t/100
#     t_list.append(threshold)
#     if cfg.METRIC.TYPE == 'pedestrian':
#         valid_result = get_pedestrian_metrics(gt_label, preds_probs,threshold=threshold)
#         valid_map, _ = get_map_metrics(gt_label, preds_probs)

#             # print(f'Evaluation on test set, \n',
#             #     'ma: {:.4f},  map: {:.4f}, label_f1: {:4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
#             #         valid_result.ma, valid_map, np.mean(valid_result.label_f1), np.mean(valid_result.label_pos_recall),
#             #         np.mean(valid_result.label_neg_recall)),
#             #     'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
#             #         valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
#             #         valid_result.instance_f1)
#             #     )
#         MA_list.append(valid_result.ma)
#         ACC_list.append(valid_result.instance_acc)
#         PRE_list.append(valid_result.instance_prec)
#         REC_list.append(valid_result.instance_recall)
#         F1_list.append(valid_result.instance_f1)
# import matplotlib.pyplot as plt
# plt.plot(t_list,MA_list,label=r'MA',linewidth=1)
# plt.plot(t_list,ACC_list,label=r'ACC',linewidth=1)
# plt.plot(t_list,PRE_list,label=r'PRE',linewidth=1)
# plt.plot(t_list,REC_list,label=r'RE',linewidth=1)
# plt.plot(t_list,F1_list,label=r'F1',linewidth=1)
# plt.xlabel('Threshold')
# plt.ylabel('')
# # plt.plot(t, sol_all[i][:,2],color='b',label=r'$x_{3}(t)$',linewidth=1)
# # plt.plot(t, sol_all[i][:,3],color='y',label=r'$x_{4}(t)$',linewidth=1)
# plt.legend()
# plt.show()

        # with open(os.path.join(model_dir, 'efficientnet_v2_s_infer_rap2_pa100k_market1501_msmt17_WF3_grayimg_test.pkl'), 'wb+') as f:
        #     pickle.dump([valid_result, gt_label, preds_probs, attn_list, path_list], f, protocol=4)
    # for img in tqdm(os.listdir(img_dir)):
    #     b = [0 for i in range(41)]
    #     count+=1
    #     img_path = os.path.join(img_dir, img)
    #     image = Image.open(img_path)
    #     img_copy = image.copy()
    #     image = valid_tsfm(image).float()
    #     image = image.unsqueeze(0)
    #     image = image.cuda()
    #     valid_logits, attns = model(image)
    #     valid_probs = torch.sigmoid(valid_logits[0])
    #     pred_probs = valid_probs.cpu().numpy().tolist()[0]
    #     for idx,p in enumerate(pred_probs):
    #         if p >= 0.95:
    #             if idx == 0:
    #                 b[0] = 1
    #             elif idx == 1:
    #                 b[2] = 1
    #                 b[3] = 1
    #             elif idx == 2:
    #                 b[4] = 1
    #             elif idx == 3:
    #                 b[6] = 1
    #             elif idx == 4:
    #                 b[7] = 1
    #             elif idx == 5:
    #                 b[8] = 1
    #                 b[9] = 1
    #             elif idx == 6:
    #                 b[10] = 1
    #             elif idx == 7:
    #                 b[11] = 1
    #             elif idx == 8:
    #                 b[12] = 1
    #             elif idx == 9:
    #                 b[13] = 1
    #                 b[14] = 1
    #     for item in a:
    #         if pred_probs[item] >= 0.95:
    #             img_copy.save(os.path.join(reid_market1501,img))
    #             with open(os.path.join(label_market1501,img[:-3]+"txt"), 'w') as f:
    #                 for idx,j in enumerate(b):
    #                     f.write(str(j))
    #                     if idx != len(b)-1:
    #                         f.write(",")
    #                 f.close()
    #             break
        