# First inference the custom_demo_inference.py
# This will results in pose estimation heatmap for each frame
# Then we average the heatmap to get the heatmap_avg.pkl as the attack target to achieve temporal-consistency

import pickle
import torch

heatmap_dict = pickle.load(open('heatmap_dict.pkl', 'rb'))

heatmap_tensor = torch.cat(list(heatmap_dict.values()))
heatmap_avg = heatmap_tensor.mean(dim=0)

save_path = 'heatmap_avg.pkl' 
pickle.dump(heatmap_avg, open(save_path, 'wb'))