import logging
import numpy as np
from vissl.utils.extract_features_utils import ExtractedFeaturesLoader
from vissl.utils.misc import merge_features

from vissl.utils.logger import setup_logging, shutdown_logging

setup_logging(__name__)
logging.info(f"Logging set up")

extracted_feats_path = "/home/bbea/vissl/outputs/extracted_features/simclr_imagenet1k_resnet50_phase999_final_trunk_feats_3view"
merged_feats_path = "/home/bbea/vissl/outputs/simclr_1node_imagenet1k_resnet50_phase999_3view_embeddings_dataset/"

layer = "res5avg"
for split in ["train"]:
    merged_features = ExtractedFeaturesLoader.load_features(input_dir=extracted_feats_path, split=split, layer=layer)

    print(merged_features.keys())

    np.save(merged_feats_path + split + "_3view_embeddings.npy", merged_features['features'])
    print(merged_features['features'].shape)

    np.save(merged_feats_path + split + "_3view_targets.npy", merged_features['targets'])
    print(merged_features['targets'].shape)

logging.info("All Done!")
shutdown_logging()
