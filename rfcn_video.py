# -*- coding: utf-8 -*-
"""
rfcn-dcn-demo script
"""
import argparse
import os
import sys
import logging
import pprint
import cv2
sys.path.insert(0, os.path.join('/opt/dcn', 'rfcn'))
import _init_paths
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
import random
import urllib
import json
import copy
import time
from rfcn_dcn_config_JH_logProcess import rfcn_dcn_config as RFCN_DCN_CONFIG


def init_detect_model():
    # get symbol
    pprint.pprint(config)
    update_config(RFCN_DCN_CONFIG["config_yaml_file"])
    config.symbol = 'resnet_v1_101_rfcn_dcn'
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)
    arg_params, aux_params = load_param(
        os.path.join(RFCN_DCN_CONFIG['modelParam']['modelBasePath'], 'rfcn_voc'), RFCN_DCN_CONFIG['modelParam']['epoch'], process=True)
    return [sym, arg_params, aux_params]


def show_boxes_write_rg(fileOp=None, im=None, dets=None, classes=None, vis=None, scale=1.0, count=0):
    color_black = (0, 0, 0)
    # write to terror det rg tsv file
    # imageName = image_name
    writeInfo = []
    # "alpha%d" % = 0
    for cls_idx, cls_name in enumerate(classes[1:], start=1):
        if cls_idx not in RFCN_DCN_CONFIG['need_label_dict'].keys():
            continue
        write_bbox_info = {}
        #write_bbox_info['class'] = cls_name
        write_bbox_info['class'] = RFCN_DCN_CONFIG['need_label_dict'][cls_idx]
        """
            change log : rg : result class name :
                guns_true->guns,knives_true->knives
        """
        write_bbox_info['index'] = cls_idx
        cls_dets = dets[cls_idx-1]
        # color = (random.randint(0, 256), random.randint(
        #     0, 256), random.randint(0, 256))
        for det in cls_dets:
            bbox = det[:4] * scale
            score = det[-1]
            if float(score) < RFCN_DCN_CONFIG['need_label_thresholds'][cls_idx]:
                continue
            bbox = map(int, bbox)
            one_bbox_write = copy.deepcopy(write_bbox_info)
            bbox_position_list = []
            bbox_position_list.append([bbox[0], bbox[1]])
            bbox_position_list.append([bbox[2], bbox[1]])
            bbox_position_list.append([bbox[2], bbox[3]])
            bbox_position_list.append([bbox[0], bbox[3]])
            one_bbox_write["pts"] = bbox_position_list
            # one_bbox_write["score"] = float(score)
            one_bbox_write["score"] = float(score)
            writeInfo.append(one_bbox_write)
            if vis is not None and im is not None:
                cv2.rectangle(
                    im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color_black, thickness=3)
                cv2.putText(im, '%s %.3f' % (cls_name, score), (
                    bbox[0], bbox[1] + 15), color=color_black, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    fileOp.write("%s\t%s" % ("alpha%d" % count, json.dumps(writeInfo)))
    fileOp.write('\n')
    fileOp.flush()
    if vis is not None and im is not None:
        out_file = os.path.join(vis, "alpha%d.jpg" % count)

        cv2.imwrite(out_file, im)
        return im
    

def process_image_fun(imagesPath=None, fileOp=None, vis=None, model_params_list=None, count=0):
    # init rfcn dcn detect model (mxnet)
    # model_params_list = init_detect_model()

    # num_classes = RFCN_DCN_CONFIG['num_classes']  # 0 is background,
    classes = RFCN_DCN_CONFIG['num_classes_name_list']
    min_threshold = min(
        list(RFCN_DCN_CONFIG['need_label_thresholds'].values()))
    
    im_name = imagesPath
    all_can_read_image = []
    data = []
    all_can_read_image.append(im_name)
    target_size = config.SCALES[0][0]
    max_size = config.SCALES[0][1]
    im, im_scale = resize(im_name, target_size, max_size,
                          stride=config.network.IMAGE_STRIDE)
    im_tensor = transform(im, config.network.PIXEL_MEANS)
    im_info = np.array(
        [[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
    data.append({'data': im_tensor, 'im_info': im_info})

    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names]
            for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max(
        [v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])]
                    for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]

    predictor = Predictor(model_params_list[0], data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=model_params_list[1], aux_params=model_params_list[2])
    nms = gpu_nms_wrapper(config.TEST.NMS, 0)

    for idx, im_name in enumerate(all_can_read_image):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx, provide_data=[
                                     [(k, v.shape) for k, v in zip(data_names, data[idx])]], provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2]
                  for i in xrange(len(data_batch.data))]

        tic()
        scores, boxes, data_dict = im_detect(
            predictor, data_batch, data_names, scales, config)
        boxes = boxes[0].astype('f')
        scores = scores[0].astype('f')
        dets_nms = []
        for j in range(1, scores.shape[1]):
            cls_scores = scores[:, j, np.newaxis]
            cls_boxes = boxes[:,
                              4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets)
            cls_dets = cls_dets[keep, :]
            cls_dets = cls_dets[cls_dets[:, -1] > min_threshold, :]
            dets_nms.append(cls_dets)
        print('testing {} {:.4f}s'.format(im_name, toc()))
        show_boxes_write_rg(im=im_name, dets=dets_nms,
                   classes=classes, scale=1, vis=vis, fileOp=fileOp)
    return im
        
