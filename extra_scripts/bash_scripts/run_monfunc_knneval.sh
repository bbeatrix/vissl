#!/bin/bash

for monfunc in "pow";
do
    for k in {0.1,0.3,0.5,0.7,0.9,1.0,2.0,3.0,4.0};
    do
        echo "Processing for func $monfunc ..."
        echo "Processing for param value $k ..."

        main_config="benchmark/nearest_neighbor/eval_resnet_8gpu_in1k_kNN.yaml"
        params_file="/data/shared/data/vissl_pretrained_models/simclr_1node_imagenet1k_resnet50_ours/model_final_checkpoint_phase999.torch"
        checkpoint_dir="./outputs/knneval/monfunc_pow_grid/monfunc_${monfunc}_with_param_${k}"

        use_monfunc_layer=True
        test_only=False
        num_gpus=4

        CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/nearest_neighbor_test.py config=${main_config} config.DISTRIBUTED.NUM_PROC_PER_NODE=${num_gpus} config.TEST_ONLY=${test_only} config.MODEL.WEIGHTS_INIT.PARAMS_FILE=${params_file} config.CHECKPOINT.DIR=${checkpoint_dir} config.MODEL.TRUNK.RESNETS.USE_MONFUNC_LAYER=${use_monfunc_layer} config.MODEL.TRUNK.RESNETS.MONFUNC_CLASS=${monfunc} config.MODEL.TRUNK.RESNETS.MONFUNC_PARAM_VALUE=${k}
        echo "Done with $monfunc param $k"

    done
done