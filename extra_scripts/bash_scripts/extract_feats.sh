#!/bin/bash

SOURCE_DIR=/data/shared/data/vissl_pretrained_models
files=("$SOURCE_DIR"/simclr_1node_imagenet1k_resnet50_ours/*.torch)

echo $files

for f in ${files[@]}
do
    IFS='/' read -r -a fpath_array <<< "$f"
    IFS='.' read -r -a fname_array <<< "${fpath_array[-1]}"

    echo "Processing $f ..."

    main_config="/pretrain/simclr/simclr_1node_resnet.yaml"
    pretrained_model="${f}"
    outdir_name=${fname_array[0]}
    output_dir="${SOURCE_DIR}/extracted_features/simclr_1node_imagenet1k_resnet50_ours/${outdir_name}/"

    test_only=True
    num_gpus=8

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/run_distributed_engines.py config=${main_config} +config/feature_extraction=extract_resnet_in1k_8gpu.yaml +config/feature_extraction/trunk_only=rn50_res5.yaml config.DISTRIBUTED.NUM_PROC_PER_NODE=${num_gpus} config.TEST_MODEL=True config.TEST_ONLY=${test_only} config.EXTRACT_FEATURES.CHUNK_THRESHOLD=-1 config.DISTRIBUTED.RUN_ID="auto" config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=256 config.DATA.TEST.BATCHSIZE_PER_REPLICA=256 config.MODEL.WEIGHTS_INIT.PARAMS_FILE=${pretrained_model} config.CHECKPOINT.DIR=${output_dir}
    echo "Done with $f!"
done
