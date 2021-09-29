from detecto import core, utils, visualize
import os
import torch
import pandas
#TRAINING OF THE MODEL, PLEASE USE YOUR LABELED IMAGES FOLDER HERE
print(torch.cuda.is_available())
dataset = core.Dataset('YOUR TRAINING FOLDER HERE')
model = core.Model(['YOUR ROUTES HERE'])
model.fit(dataset)
model.save('NAME OF YOUR OUTPUT FILE.pth')

