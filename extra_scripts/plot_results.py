import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-poster")

top1_accs0 = [0.1, 40.788, 43.01, 44.24, 45.11, 45.738, 45.978, 46.42, 46.786, 47.046, 47.37, ]
top5_accs0 = [0.5, 63.488, 66.124, 67.604, 68.234, 68.864, 69.412, 69.682, 69.978, 70.246, 70.478, ]

top1_accs1 = [0.1, 47.37, 48.856, 49.8, 50.008, 50.14, 50.234, 50.458, 50.34, 50.554, 50.436, 50.436,
             50.318, 50.312, 50.278, 50.014, 49.988, 50.072, 49.914, 49.85, 49.786, 49.624,
             49.672, 49.622, ]
top5_accs1 = [0.5, 70.478, 71.92, 72.718, 73.146, 73.346, 73.306, 73.398, 73.482, 73.36, 73.454, 73.33,
             73.34, 73.236, 73.01, 73.238, 73.042, 73.11, 72.89, 72.806, 72.938, 72.712,
             72.842, 72.918, ]

top1_accs2 = [49.628, 49.706, 49.628, 49.614, 49.584, 49.608, 49.512, ]
top5_accs2 = [72.79, 72.808, 72.806, 72.716, 72.846, 72.76, 72.834, ]

x0 = np.arange(0.0, 1.1, 0.1)
x1 = np.arange(2, 31)

x = list(x0) + list(x1)
top1_accs = top1_accs0 + top1_accs1[2:] + top1_accs2
top5_accs = top5_accs0 + top5_accs1[2:] + top5_accs2

plt0 = plt.figure(figsize=(15, 9))

p1 = plt.plot(x, top1_accs, 'o-', label="top1 test acc")
p2 = plt.plot(x, top5_accs, 'o-', label="top5 test acc")
plt.plot(x, [61.28] * len(x), '--', alpha=0.4, color=p1[0].get_color(), label="orig top1 test acc")
plt.plot(x, [81.18] * len(x), '--', alpha=0.4, color=p2[0].get_color(), label="orig top1 test acc")

plt.xlabel('Sign alfa value')
plt.ylabel('Imagenet test accuracy')
plt.grid(True, color="0.9")
plt.legend()
plt.title('Imagenet final checkpoint knn eval after trunk + sign function')
plt.tight_layout()
plt.savefig('outputs/plot.png', bbox_inches='tight')