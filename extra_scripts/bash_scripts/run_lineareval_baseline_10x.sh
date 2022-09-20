#!/bin/bash

for k in {1..10..1};
do
    echo "Processing for seed value $k ..."

    main_config="benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear_baseline_bs512.yaml"
    params_file="/data/shared/data/vissl_pretrained_models/simclr_1node_imagenet1k_resnet50_ours/model_final_checkpoint_phase999.torch"
    checkpoint_dir="/home/bbea/outputs/lineareval_baseline_10xseed/lineareval_baseline_with_seed_${k}"

    num_gpus=7

    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tools/run_distributed_engines.py config=${main_config} config.SEED_VALUE=$k config.DISTRIBUTED.NUM_PROC_PER_NODE=${num_gpus} config.MODEL.SYNC_BN_CONFIG.GROUP_SIZE=${num_gpus} config.MODEL.WEIGHTS_INIT.PARAMS_FILE=${params_file} config.CHECKPOINT.DIR=${checkpoint_dir} config.HOOKS.TENSORBOARD_SETUP.LOG_DIR=${checkpoint_dir}
    
    echo "Done with seed $k"
done
