dataset_type = 'CocoDataset'

clearml_dataset_id = "845004a71fe84aee932c4c2efbe14799"
data_root = "/clearml_agent_cache/storage_manager/datasets/ds_" + clearml_dataset_id + "/FoodSeg103"
# data_root = "/Eren/Data/LogMeal/FoodSeg103"

pretrained_clearml_dataset_id = "f965f0b463e74e999d11fa45f3963870"
pretrained_data_path = "/clearml_agent_cache/storage_manager/datasets/ds_" + pretrained_clearml_dataset_id +\
                       "swin_small_patch4_window7_224.pth"


project_name = "Food_Instance_Segmentation_FoodSeg103"
experiment_name = "Swin_SCNet_Raw"
group_name = None
wandb_id = None  # '1224ilu3'
wandb_resume = True

work_dir = '/Eren/Data/LogMeal/Workdir/' + experiment_name

log_level = 'INFO'
load_from = None
resume_from = None
# resume_from = work_dir +"/latest.pth"
workflow = [('train', 1), ('val', 1)]

valid_classes = ['background',
 'candy',
 'egg',
 'french',
 'chocolate',
 'biscuit',
 'popcorn',
 'pudding',
 'ice',
 'cheese',
 'cake',
 'wine',
 'milkshake',
 'coffee',
 'juice',
 'milk',
 'tea',
 'almond',
 'red',
 'cashew',
 'dried',
 'soy',
 'walnut',
 'peanut',
 'egg',
 'apple',
 'date',
 'apricot',
 'avocado',
 'banana',
 'strawberry',
 'cherry',
 'blueberry',
 'raspberry',
 'mango',
 'olives',
 'peach',
 'lemon',
 'pear',
 'fig',
 'pineapple',
 'grape',
 'kiwi',
 'melon',
 'orange',
 'watermelon',
 'steak',
 'pork',
 'chicken',
 'sausage',
 'fried',
 'lamb',
 'sauce',
 'crab',
 'fish',
 'shellfish',
 'shrimp',
 'soup',
 'bread',
 'corn',
 'hamburg',
 'pizza',
 'hanamaki',
 'wonton',
 'pasta',
 'noodles',
 'rice',
 'pie',
 'tofu',
 'eggplant',
 'potato',
 'garlic',
 'cauliflower',
 'tomato',
 'kelp',
 'seaweed',
 'spring',
 'rape',
 'ginger',
 'okra',
 'lettuce',
 'pumpkin',
 'cucumber',
 'white',
 'carrot',
 'asparagus',
 'bamboo',
 'broccoli',
 'celery',
 'cilantro',
 'snow',
 'cabbage',
 'bean',
 'onion',
 'pepper',
 'green',
 'French',
 'king',
 'shiitake',
 'enoki',
 'oyster',
 'white',
 'salad',
 'other']


classes = valid_classes
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='AutoAugment',
        policies=[[{
            'type': 'Translate',
            'prob': 0.5,
            'level': 6
        }, {
            'type': 'Shear',
            'prob': 0.5,
            'level': 6
        }, {
            'type': 'ContrastTransform',
            'prob': 0.3,
            'level': 4
        }, {
            'type': 'ColorTransform',
            'prob': 0.5,
            'level': 7
        }],
                  [{
                      'type': 'Rotate',
                      'prob': 0.5,
                      'level': 6
                  }, {
                      'type': 'Translate',
                      'prob': 0.5,
                      'level': 6
                  }, {
                      'type': 'EqualizeTransform',
                      'prob': 0.3
                  }, {
                      'type': 'BrightnessTransform',
                      'prob': 0.3,
                      'level': 6
                  }]]),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.01,
        dataset=dict(
            type='CocoDataset',
            classes=valid_classes,
            ann_file=data_root + '/foodseg103_train.json',
            img_prefix=data_root + '/Images/img_dir/train/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    type='AutoAugment',
                    policies=[[{
                        'type': 'Translate',
                        'prob': 0.5,
                        'level': 6
                    }, {
                        'type': 'Shear',
                        'prob': 0.5,
                        'level': 6
                    }, {
                        'type': 'ContrastTransform',
                        'prob': 0.3,
                        'level': 4
                    }, {
                        'type': 'ColorTransform',
                        'prob': 0.5,
                        'level': 7
                    }],
                              [{
                                  'type': 'Rotate',
                                  'prob': 0.5,
                                  'level': 6
                              }, {
                                  'type': 'Translate',
                                  'prob': 0.5,
                                  'level': 6
                              }, {
                                  'type': 'EqualizeTransform',
                                  'prob': 0.3
                              }, {
                                  'type': 'BrightnessTransform',
                                  'prob': 0.3,
                                  'level': 6
                              }]]),
                dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='AutoAugment',
                policies=[[{
                    'type': 'Translate',
                    'prob': 0.5,
                    'level': 6
                }, {
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 6
                }, {
                    'type': 'ContrastTransform',
                    'prob': 0.3,
                    'level': 4
                }, {
                    'type': 'ColorTransform',
                    'prob': 0.5,
                    'level': 7
                }],
                          [{
                              'type': 'Rotate',
                              'prob': 0.5,
                              'level': 6
                          }, {
                              'type': 'Translate',
                              'prob': 0.5,
                              'level': 6
                          }, {
                              'type': 'EqualizeTransform',
                              'prob': 0.3
                          }, {
                              'type': 'BrightnessTransform',
                              'prob': 0.3,
                              'level': 6
                          }]]),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=valid_classes,
        ann_file=data_root + '/foodseg103_test.json',
        img_prefix=data_root + '/Images/img_dir/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=valid_classes,
        ann_file=data_root + '/foodseg103_test.json',
        img_prefix=data_root + '/Images/img_dir/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(metric=['bbox', 'segm'])
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    periods=[6, 6, 6, 6, 10, 10],
    restart_weights=[1, 0.8, 0.6, 0.4, 0.3, 0.2],
    min_lr=1e-05)
total_epochs = 43
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook'),
           # dict(type='TensorboardLoggerHook'),
            dict(type='WandbLoggerHook', init_kwargs=dict(
                project=project_name,
                name=experiment_name, group=group_name, resume=wandb_resume, id=wandb_id), with_step=False)
           ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')

model = dict(
    type='SCNet',
    pretrained=pretrained_data_path,
    backbone=dict(
        type='SwinTransformer',
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=True
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 384 * 2],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='SCNetRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='SCNetBBoxHead',
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=len(valid_classes),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='SCNetBBoxHead',
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=len(valid_classes),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='SCNetBBoxHead',
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=len(valid_classes),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='SCNetMaskHead',
            num_convs=12,
            in_channels=256,
            conv_out_channels=256,
            num_classes=len(valid_classes),
            conv_to_res=True,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        glbctx_head=dict(
            type='GlobalContextHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=len(valid_classes),
            loss_weight=3.0,
            conv_to_res=True),
        feat_relay_head=dict(
            type='FeatureRelayHead',
            in_channels=1024,
            out_conv_channels=256,
            roi_feat_size=7,
            scale_factor=2)),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
                gpu_assign_thr=2),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1,
                    gpu_assign_thr=2),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    ignore_iof_thr=-1,
                    gpu_assign_thr=2),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    ignore_iof_thr=-1,
                    gpu_assign_thr=2),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
gpu_ids = range(0, 1)
