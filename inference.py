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

class TestModel():
    def __init__(self, model):
        self.model = model
        
    def predict(self, img_path):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        SR_dir = join(os.path.dirname(img_path), 'result')  #--------------------------SR results save path
        if not os.path.exists(SR_dir):
            os.makedirs(SR_dir)

        testloader = DataLoader(DataValSet(img_path), batch_size=1, shuffle=False, pin_memory=False)
        model = torch.load(self.model)
        model = model.to(device)

        with torch.no_grad():
            for iteration, batch in enumerate(testloader, 1):
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