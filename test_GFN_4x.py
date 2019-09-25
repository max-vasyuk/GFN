from __future__ import print_function
import os
from os.path import join
from torchvision import transforms
from torchvision import utils as utils
import torch
from torch.utils.data import DataLoader
from datasets.dataset_hf5 import DataValSet
import statistics
import matplotlib.pyplot as plot
import re

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(test_gen, model, SR_dir):
    model = model.to(device)
    with torch.no_grad():
        for iteration, batch in enumerate(test_gen, 1):
            LR_Blur = batch[0]
            HR = batch[1]
            LR_Blur = LR_Blur.to(device)
            HR = HR.to(device)
            test_Tensor = torch.cuda.FloatTensor().resize_(1).zero_() + 1
            gated_Tensor = torch.cuda.FloatTensor().resize_(1).zero_() + 1


            [lr_deblur, sr] = model(LR_Blur, gated_Tensor, test_Tensor)
            sr = torch.clamp(sr, min=0, max=1)
            torch.cuda.synchronize()#wait for CPU & GPU time syn
            resultSRDeblur = transforms.ToPILImage()(sr.cpu()[0])
            resultSRDeblur.save(join(SR_dir, 'result.png'))

root_val_dir = '/content/68314_v5m55_a0.jpeg'
SR_dir = join(os.path.dirname(root_val_dir), 'result')  #--------------------------SR results save path

if not os.path.exists(SR_dir):
    os.makedirs(SR_dir)

testloader = DataLoader(DataValSet(root_val_dir), batch_size=1, shuffle=False, pin_memory=False)

model_dir = '/content/GFN/models/model_gfn_24_09_(09:59).pkl'
model = torch.load(model_dir)
test(testloader, model, SR_dir)