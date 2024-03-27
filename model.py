import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class SimCLR(nn.Module):

    def __init__(self, resnet=18, out_dim=256, projection="nonlinear"):
        super(SimCLR, self).__init__()
        if resnet == 18:
            self.resnet = torchvision.models.resnet18(weights=None)
            self.enc_out_dim = 512
        elif resnet == 50:
            self.resnet = torchvision.models.resnet50(weights=None)
            self.enc_out_dim = 2048
        else:
            raise ValueError("resnet should be 18 or 50.")

        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        if projection == "nonlinear":
            self.projection = nn.Sequential(
                nn.Linear(self.enc_out_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, out_dim)
            )
        elif projection == "linear":
            self.projection = nn.Linear(512, out_dim)
        elif projection == "identity":
            self.projection = nn.Identity()
        else:
            raise ValueError("projection should be nonlinear, linear, or identity.")
    
    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)

        z_i = self.projection(h_i)
        z_j = self.projection(h_j)

        return h_i, h_j, z_i, z_j



class NTXentLoss(nn.Module):
    def __init__(self, batch_size, device, t=0.5):
        super(NTXentLoss, self).__init__()
        self.t = t
        # moving targets to device once to avoid overhead
        # requires all batches to be of same size
        targets = torch.arange(2*batch_size)
        targets = torch.cat([targets[batch_size:], targets[:batch_size]]).long()
        self.targets = targets.to(device) 

    def forward(self, z_i, z_j):
        x = torch.cat([z_i, z_j], dim=0)
        x = F.normalize(x, dim=1)
        x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
        x_scale = x_scores / self.t   # scale with temperature

        # Set the diagonals to be large negative values, which become zeros after softmax.
        x_scale.fill_diagonal_(-1e5)

        # targets 2N elements.
        return F.cross_entropy(x_scale, self.targets)