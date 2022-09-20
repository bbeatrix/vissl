import numpy as np
import matplotlib.pyplot as plt
import re

plt.style.use("seaborn-poster")

output_file = "/home/bbea/vissl/outputs/wr_replaced_fixedmlp_weight_interpolation_eval/grid.out"
with open(output_file) as f:
    output_contents = f.read()

acc_pattern = re.compile(r"name: test_accuracy_list_meter, value: {'top_1': {0: (\d+\.?\d+)}, 'top_5': {0: (\d+\.?\d+)}}")
acc_search = re.findall(acc_pattern, output_contents)

if acc_search:
    top1_accs = [np.float(acc_search[i][0]) for i in range(len(acc_search))]
    top5_accs = [np.float(acc_search[i][1]) for i in range(len(acc_search))]

x = np.arange(0.0, 2.002, 0.002)

plt0 = plt.figure(figsize=(15, 9))

p1 = plt.plot(x, top1_accs, '-', label="top1 test acc")
p2 = plt.plot(x, top5_accs, '-', label="top5 test acc")
plt.plot(x, [68.6566] * len(x), '--', alpha=0.4, color=p1[0].get_color(), label="baseline top1 test acc")
plt.plot(x, [89.0942] * len(x), '--', alpha=0.4, color=p2[0].get_color(), label="baseline top5 test acc")

plt.xlabel('Relu weight value')
plt.ylabel('Imagenet test accuracy')
plt.grid(True, color="0.9")
plt.legend()
plt.title('Imagenet final checkpoint linear eval')
plt.tight_layout()
plt.savefig('/home/bbea/vissl/outputs/grid_result_plot.png', bbox_inches='tight')