import imageio
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import torch
import yaml
import torch.nn as nn

plt.style.use("seaborn-poster")

dirs_path = "/home/bbea/vissl/outputs/lineareval/wr_replaced_multisx_again/"
config_file = os.path.join(dirs_path, "train_config.yaml")
with open(config_file, 'r') as stream:
    parsed_yaml=yaml.safe_load(stream)
shifts_linspace_params = parsed_yaml['MODEL']['TRUNK']['RESNETS']['SHIFTED_RELUS_WEIGHTED_SUM']['shifts_linspace']

print(shifts_linspace_params)
shifts = np.linspace(*shifts_linspace_params)
print(shifts)

dirs_content = os.listdir(dirs_path)
phase_checkpoints = [f for f in dirs_content if "model_phase" in f]
#print("Number of phase checkpoints: ", len(phase_checkpoints))
#print(phase_checkpoints)

def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

phase_checkpoints.sort(key=num_sort)

def shifted_relus_weighted_sum(shifts, learned_weights, x):
    output = 0
    for i in range(len(shifts)):
        output += learned_weights[i] * np.maximum(0, x - shifts[i])
    return output

class MagicActivation(nn.Module):
    """
    This module can be used to attach a layer that  shifts relus with different scalars, and returns 
    the weighted sum of those applied on the input.

    Accepts a 2D input tensor. Also accepts 4D input tensor of shape `N x C x 1 x 1`.
    """

    def __init__(
        self,
        model_config: AttrDict,
    ):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
        """
        super().__init__()
        # err_message = "Last Relu should be removed when using ShiftedRelusWeightedSum layer" 
        # assert model_config.TRUNK.RESNETS.REMOVE_LAST_RELU == True, err_message  
        if model_config.TRUNK.RESNETS.REMOVE_LAST_RELU == False:
            logging.info("Last relu in trunk is not removed, applying shift")

        hidden_dim = model_config.TRUNK.RESNETS.get("MAGIC_ACTIVATION_HIDDEN_DIM", 10)

        layer1 = torch.nn.Linear(1, hidden_dim)
        layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        layer3 = torch.nn.Linear(hidden_dim, 1)
        bn = torch.nn.BatchNorm1d(num_features=1)
        self.do_not_freeze_layers = nn.ModuleList([bn, layer1, layer2, layer3])

        

    def forward(self, batch: torch.Tensor):
        """
        Args:
            batch (torch.Tensor): 2D torch tensor or 4D tensor of shape `N x C x 1 x 1`
        Returns:
            out (torch.Tensor): 2D output torch tensor
        """
        if isinstance(batch, list):
            assert (
                len(batch) == 1
            ), "ShiftedRelusWeightedSum input should be either a tensor (2D, 4D) or list containing 1 tensor."
            batch = batch[0]
        #if batch.ndim > 2:
        #    assert all(
        #        d == 1 for d in batch.shape[2:]
        #    ), f"ShiftedRelusWeightedSum expected 2D input tensor or 4D tensor of shape NxCx1x1. got: {batch.shape}"
        #    batch = batch.reshape((batch.size(0), batch.size(1)))

        out = batch.view(-1)
        out = out.unsqueeze(dim=1)
        for i, layer in enumerate(self.do_not_freeze_layers):
            out = layer(out)
            if i < 2:
                out = torch.nn.functional.tanh(out)
        out = out.view(batch.shape)
        return out


def plot_shifted_relus_weighted_sum(shifts, learned_weights, name):
    x = np.linspace(-10, 30, 100)
    f_x = shifted_relus_weighted_sum(shifts, learned_weights, x)

    plt.figure(figsize=(15, 9))

    plt.plot(x, f_x, '-', label="learned function")

    plt.plot(x, np.maximum(0, x), '--', alpha=0.4, color="orange", label="relu")

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, color="0.9")
    plt.legend()
    plt.title(f'shifted relus weighted sum function at: {name}')
    plt.tight_layout()
    save_filename = dirs_path + name + "_learned_func.png"
    plt.savefig(save_filename, bbox_inches='tight')
    plt.clf()
    plt.cla()
    return save_filename

filenames = []
for checkpoint_file in phase_checkpoints: 
    print(checkpoint_file)
    phase_checkpoint_dict = torch.load(os.path.join(dirs_path, checkpoint_file))
    learned_weights = phase_checkpoint_dict['classy_state_dict']['base_model']['model']['trunk']['_feature_blocks.layer4.2.relu.weights'].cpu().numpy()
     #['heads']['0.shifted_relus_weighted_sum.weights'].cpu().numpy()
    print("Weights at phase: ", learned_weights)
    name = checkpoint_file.split(".")[0].split("_")[1] 
    save_filename = plot_shifted_relus_weighted_sum(shifts, learned_weights, name)
    for i in range(10):
        filenames.append(save_filename)

# build gif
save_gif_path = dirs_path + dirs_path.split("/")[-2] + '_learned_func.gif'
print("Saving plot gif to ", save_gif_path)
with imageio.get_writer(save_gif_path, mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
print('Removing images')
# Remove files
for filename in set(filenames):
    os.remove(filename)
print('FIN')

