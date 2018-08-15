rfcn_dcn_config = {
    "config_yaml_file": "/workspace/inference/models/resnet_v1_101_terror_dcn_rfcn_end2end_ohem.yaml",
    "modelParam": {
        "modelBasePath":"/workspace/inference/models",
        "epoch":14
    },
    "one_batch_size":200,
    'num_classes':16,
    'num_classes_name_list': ['__background__',
                              'islamic flag', 'isis flag', 'tibetan flag', 'knives_true', 'guns_true',
                              'knives_false', 'knives_kitchen',
                              'guns_anime', 'guns_tools', 'BK_LOGO_1',
                              'BK_LOGO_2', 'BK_LOGO_3', 'BK_LOGO_4',
                              'BK_LOGO_5', 'BK_LOGO_6',
                              'not terror'],
    'need_label_dict': {
        1: 'islamic flag',
        2: 'isis flag',
        3: 'tibetan flag',
        4: 'knives',
        5: 'guns',
        6: 'knives_false',
        7: 'knives_kitchen',
        8: 'guns_anime',
        9: 'guns_tools',
        10: 'BK_LOGO_1',
        11: 'BK_LOGO_2',
        12: 'BK_LOGO_3',
        13: 'BK_LOGO_4',
        14: 'BK_LOGO_5',
        15: 'BK_LOGO_6',
        16: 'not terror'
    },
    'need_label_thresholds': {
        1: 0.8,
        2: 0.8,
        3: 0.8,
        4: 0.7,
        5: 0.7,
        6: 0.6,
        7: 0.6,
        8: 0.6,
        9: 0.6,
        10:0.5,
        11:0.5,
        12:0.5,
        13:0.5,
        14:0.5,
        15:0.5,
        16: 1.0
    }
}

