import numpy as np
from vissl.utils.extract_features_utils import ExtractedFeaturesLoader
from vissl.utils.misc import merge_features


extracted_feats_path = "/home/bbea/vissl/outputs/extracted_features/simclr_1node_imagenet1k_resnet50_phase999_final_trunkfeats"
merged_feats_path = "/home/bbea/vissl/outputs/simclr_1node_imagenet1k_resnet50_phase999_embeddings_dataset/"

layer = "res5avg"
for split in ["train", "test"]: 
    merged_features = ExtractedFeaturesLoader.load_features(input_dir=extracted_feats_path, split=split, layer=layer)

    print(merged_features.keys())

    np.save(merged_feats_path + split + "_embeddings.npy", merged_features['features'])
    print(merged_features['features'].shape)

    np.save(merged_feats_path + split + "_targets.npy", merged_features['targets'])
    print(merged_features['targets'].shape)
