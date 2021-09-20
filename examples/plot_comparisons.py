import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import numpy as np
import glob
import matplotlib.gridspec as gridspec


random_agent = glob.glob("plots/fishing_agent_random*")
random_agent.sort()
single_agent = glob.glob("plots/sac_fishingv1_sigma0*")
single_agent.sort()
n_rows = len(random_agent)

for row in range(n_rows):
  fig, axs = plt.subplots(1, 2, figsize=(10,10))
  for col in range(2):
    if col == 0:
      img = mpimg.imread(random_agent[row])
    else:
      img = mpimg.imread(single_agent[row])
    axs[col].imshow(img)
  plt.tight_layout()
  plt.savefig(f"plots/trash_comparison_{row}.png")
