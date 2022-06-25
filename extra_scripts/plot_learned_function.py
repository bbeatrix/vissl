import imageio
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import torch
import yaml

plt.style.use("seaborn-poster")

dirs_path = "/home/bbea/vissl/outputs/_on_embeddings/lineareval_shiftedrelusweightedsum_wlinear_without_relu/"
config_file = os.path.join(dirs_path, "train_config.yaml")
with open(config_file, 'r') as stream:
    parsed_yaml=yaml.safe_load(stream)
shifts_linspace_params = parsed_yaml['MODEL']['HEAD']['PARAMS'][0][1]['shifts_linspace']

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
    learned_weights = phase_checkpoint_dict['classy_state_dict']['base_model']['model']['heads']['0.shifted_relus_weighted_sum.weights'].cpu().numpy()
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

