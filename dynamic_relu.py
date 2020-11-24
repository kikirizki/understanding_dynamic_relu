import torch
import torch.nn as nn


class Dynamic_ReLUB2D(nn.Module):
    def __init__(self, channels, reduction=4, k=2):
        super(Dynamic_ReLUB2D, self).__init__()
        self.channels = channels
        self.k = k
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2 * k * channels)
        self.sigmoid = nn.Sigmoid()
        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())

    def compute_theta(self, x):
        theta = torch.mean(x, dim=-1)
        theta = torch.mean(theta, dim=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.compute_theta(x)
        relu_coefs = theta.view(-1, self.channels, 2 * self.k) * self.lambdas + self.init_v
        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
        result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        return result
