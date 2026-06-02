import torch
from torch import nn
import numpy as np
import math

def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    if freq_init == "random":
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":
        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) / (frequency_num * 1.0 - 1))
        timescales = min_radius * np.exp(np.arange(frequency_num).astype(float) * log_timescale_increment)
        freq_list = 1.0 / timescales
    return freq_list

class Theory(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """

    def __init__(self, coord_dim=2, frequency_num=16,
                 max_radius=10000, min_radius=1000, freq_init="geometric"):
        super(Theory, self).__init__()
        self.frequency_num = frequency_num
        self.coord_dim = coord_dim
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.freq_init = freq_init
        self.embedding_dim = int(2 * 3 * frequency_num)

        freq_list = _cal_freq_list(freq_init, frequency_num, max_radius, min_radius)
        # freq_mat: (frequency_num, 6) — repeated for 3 unit vecs × sin/cos
        freq_mat = np.repeat(np.expand_dims(freq_list, axis=1), 6, axis=1)

        # unit vectors 120° apart
        uv1 = np.asarray([1.0, 0.0])
        uv2 = np.asarray([-1.0 / 2.0, math.sqrt(3) / 2.0])
        uv3 = np.asarray([-1.0 / 2.0, -math.sqrt(3) / 2.0])
        unit_vecs = np.stack([uv1, uv2, uv3], axis=0)  # (3, 2)

        self.register_buffer('freq_mat', torch.tensor(freq_mat, dtype=torch.float64))
        self.register_buffer('unit_vecs', torch.tensor(unit_vecs, dtype=torch.float64))

    def cal_embedding_dim(self):
        return self.embedding_dim

    def forward(self, coords):
        # coords: (N, 2)
        N = coords.size(0)

        # (N, 3) — dot product with each unit vector
        angles = coords @ self.unit_vecs.T

        # (N, 3, 1) → interleave sin/cos pairs → (N, 6)
        angles = angles.unsqueeze(-1).repeat(1, 1, 2).reshape(N, 6)

        # (N, 1, 6) * (frequency_num, 6) → (N, frequency_num, 6)
        angle_mat = angles.unsqueeze(1) * self.freq_mat.unsqueeze(0)

        # (N, frequency_num * 6)
        spr_embeds = angle_mat.reshape(N, -1)

        spr_embeds[..., 0::2] = torch.sin(spr_embeds[..., 0::2])
        spr_embeds[..., 1::2] = torch.cos(spr_embeds[..., 1::2])

        return spr_embeds
