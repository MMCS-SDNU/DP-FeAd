import numpy as np
import torch
import torch.nn as nn
from config import opt

import torch.autograd as autograd
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.bit+opt.num_class, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

Tensor = torch.cuda.FloatTensor if opt.use_gpu else torch.FloatTensor
def create_input_data(data):
    input_data = torch.Tensor([])
    if type(data) == list:
        input_data = torch.zeros(1, opt.bit+opt.num_class)
        return input_data
    else:
        num_keys = len(data)
        for key, value in data.items():
            key_one_hot = np.zeros(num_keys)
            key_one_hot[key] = 1
            key_one_hot = torch.tensor(key_one_hot, dtype=torch.float32)
            if isinstance(value[0], list):
                for features in value:
                    features = torch.tensor(features, dtype=torch.float32)
                    # features = torch.sign(features)
                    if opt.use_gpu:
                        features = features.cuda()
                        key_one_hot = key_one_hot.cuda()
                        input_data = input_data.cuda()
                    input_vector = torch.cat((features, key_one_hot), dim=0)
                    input_data = torch.cat((input_data, input_vector.unsqueeze(0)), dim=0)
            else:
                features = torch.tensor(value[0], dtype=torch.float32)
                # features = torch.sign(features)
                if opt.use_gpu:
                    features = features.cuda()
                    key_one_hot = key_one_hot.cuda()
                    input_data = input_data.cuda()
                input_vector = torch.cat((features, key_one_hot), dim=0)
                input_data = torch.cat((input_data, input_vector.unsqueeze(0)), dim=0)
        return input_data