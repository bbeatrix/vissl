from vissl.utils.extract_features_utils import ExtractedFeaturesLoader
from vissl.losses.barlow_twins_loss import BarlowTwinsCriterion
import os
import torch

batch_size = 512
dirs_path = "/data/shared/data/vissl_pretrained_models/extracted_features/simclr_1node_imagenet1k_resnet50_ours/"
checkpoint_dirs = os.listdir(dirs_path)
print("Number of checkpoints: ", len(checkpoint_dirs))
iteration_checkpoint_dirs = [f for f in checkpoint_dirs if "model_iteration" in f]
print("Number of iteration checkpoints: ", len(iteration_checkpoint_dirs))

test_avg_loss = []
test_avg_on_diag_loss_term = []
test_avg_scaled_off_diag_loss_term = []

for checkpoint_dir in iteration_checkpoint_dirs: 
    features = ExtractedFeaturesLoader.load_features(input_dir=os.path.join(dirs_path, checkpoint_dirs[0]), split="test", layer="res5avg")
    features_array = features['features'].reshape(-1, 2048)

    num_feats = len(features_array)
    num_batches = num_feats // batch_size

    loss_sum, on_diag_term_sum, off_diag_term_sum = 0, 0, 0
    for i in range(0, num_feats, batch_size):
        if len(features_array[i:]) < batch_size:
            break
        features_batch = features_array[i:i+batch_size]

        bt_criterion = BarlowTwinsCriterion(lambda_=0.0051,
                                            scale_loss=0.024,
                                            embedding_dim=2048)

        batch_loss, on_diag_term, scaled_off_diag_term = bt_criterion.forward_with_details(embedding=torch.from_numpy(features_batch))
        print("loss on one batch, terms: ", batch_loss, on_diag_term, scaled_off_diag_term)
        loss_sum += batch_loss
        on_diag_term_sum += on_diag_term
        off_diag_term_sum += scaled_off_diag_term

    test_avg_loss.append(loss_sum/num_batches)
    test_avg_on_diag_loss_term.append(on_diag_term_sum/num_batches)
    test_avg_scaled_off_diag_loss_term.append(off_diag_term_sum/num_batches)

print("test_avg_losses: ", test_avg_loss)
print("test_avg_on_diag_loss_terms: ", test_avg_on_diag_loss_term)
print("test_avg_scaled_off_diag_loss_term: ", test_avg_scaled_off_diag_loss_term)