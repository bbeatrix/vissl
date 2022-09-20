#!/bin/bash

i=0
for k in $(seq 0 0.002 2); 
do
    ((i=i+1))
    echo "$i. Processing for weight value $k ..."

    main_config="benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_wr_replaced_fixedmlp_constweightinit_test.yaml"
    params_file="/data/shared/data/vissl_pretrained_models/simclr_1node_imagenet1k_resnet50_ours/model_final_checkpoint_phase999.torch"
    
    checkpoint_dir="./outputs/wr_replaced_fixedmlp_weight_interpolation_eval/weight_init_with_param_${k}"

    test_only=True
    num_gpus=7

    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tools/run_distributed_engines.py config=${main_config} config.MODEL.WEIGHTS_INIT.PARAMS_FILE=${params_file} config.DISTRIBUTED.NUM_PROC_PER_NODE=${num_gpus} config.MODEL.SYNC_BN_CONFIG.GROUP_SIZE=${num_gpus} config.CHECKPOINT.DIR=${checkpoint_dir} config.TEST_ONLY=${test_only} config.MODEL.TRUNK.RESNETS.SHIFTED_RELUS_WEIGHTED_SUM.weight_init_const_value=${k}
    echo "Done with param $k"
done