#!/bin/bash

for k in {0.0,1.e-12};
do
    echo "Processing for param value $k ..."

    main_config="benchmark/nearest_neighbor/eval_resnet_8gpu_in1k_kNN.yaml"
    params_file="/data/shared/data/vissl_pretrained_models/simclr_1node_imagenet1k_resnet50_ours/model_final_checkpoint_phase999.torch"
    checkpoint_dir="./outputs/knneval/sign_grid_check/sign_with_param_${k}"

    use_sign_layer=True
    test_only=False
    num_gpus=6

    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python tools/nearest_neighbor_test.py config=${main_config} config.DISTRIBUTED.NUM_PROC_PER_NODE=${num_gpus} config.TEST_ONLY=${test_only} config.MODEL.WEIGHTS_INIT.PARAMS_FILE=${params_file} config.CHECKPOINT.DIR=${checkpoint_dir} config.MODEL.TRUNK.RESNETS.USE_SIGN_LAYER=${use_sign_layer} config.MODEL.TRUNK.RESNETS.SIGN_PARAM_VALUE=${k}
    echo "Done with $k!"

done
