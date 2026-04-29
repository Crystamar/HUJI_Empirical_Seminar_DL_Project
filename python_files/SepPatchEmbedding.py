import torch
import torch.nn as nn


class SepPatchEmbeddings(nn.Module):
    """
    An embedding class that divides an image to, biased/unbiased patches, and embeds each one with a seperate linear layer
    Assumes image dimensions: (3, 48, 48)
    """

    def __init__(self, patch_sizes, d):
        """
        :param d: The embedding dimension
        """
        super().__init__()
        self.d = d
        self.N = len(patch_sizes) ** 2
        self.s = len(patch_sizes)
        self.patches = patch_sizes
        self.embed = []
        for i in patch_sizes:
            for j in patch_sizes:
                self.embed.append(nn.Linear(i * j * 3, d))
        self.ticks = [patch_sizes[0]]
        for patch in patch_sizes[1:]:
            self.ticks.append(self.ticks[-1] + patch)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images
        :return: Embedded patches of inputs
        """
        patches = []
        for idx1, i in enumerate(self.ticks):
            for idx2, j in enumerate(self.ticks):
                p = x[:, :, i - self.patches[idx1]:i, j - self.patches[idx2]:j]
                p = torch.flatten(p, 1)
                p = self.embed[idx1 * self.s + idx2](p)
                p = torch.unsqueeze(p, 1)
                patches.append(p)
        x = torch.concat(patches, 1)
        return x
