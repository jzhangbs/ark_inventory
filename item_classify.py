import torch
import torch.nn as nn
import numpy as np
import json


with open('assets/name2idx.json') as f:
    name2idx = json.load(f)
with open('assets/idx2name.json') as f:
    idx2name = json.load(f)
NUM_CLASS = len(idx2name)


def convbr(in_c, out_c, kernel_size, stride):
    return nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size, stride, kernel_size//2, bias=False), nn.BatchNorm2d(out_c), nn.ReLU())


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            convbr(3, 16, 3, 1),
            convbr(16, 32, 3, 2),    # 64
            convbr(32, 32, 3, 1),
            convbr(32, 64, 3, 2),    # 32
            convbr(64, 64, 3, 1),
            convbr(64, 128, 3, 2),   # 16
            convbr(128, 128, 3, 1),
            convbr(128, 256, 3, 2),  # 8
            convbr(256, 256, 3, 1),
            nn.Conv2d(256, NUM_CLASS, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        out = self.model(x)
        out = out.mean((2, 3))
        return out


model = Model()
model.load_state_dict(torch.load('assets/model.bin'))
model.cuda()
model.eval()


def predict(roi_list):
    """
    Image size of 720p is recommended.
    """
    roi_np = np.stack(roi_list, 0)
    roi_t = torch.from_numpy(roi_np).float().cuda()
    with torch.no_grad():
        score = model(roi_t)
        probs = nn.Softmax(1)(score)
        predicts = score.argmax(1)

    probs = probs.cpu().data.numpy()
    predicts = predicts.cpu().data.numpy()
    return [idx2name[p] for p in predicts], [probs[i, predicts[i]] for i in range(len(roi_list))]
