import torch
import torch.nn as nn

import inspect
import sys


class SquarePatchEmbeddings(nn.Module):
    """
    Splits an image into square patches, and linearly embeds each of them
    """

    def __init__(self, patch_size, img_h, img_w, img_c=3, d=768):
        """
        :param patch_size: The size (h or w) of the patches
        :param img_h: The height of the input images
        :param img_w: The width of the input images
        :param img_c: The channels num of the input images
        :param d: The embedding dimension
        """
        if img_h < patch_size or img_w < patch_size:
            raise Exception("Image dimensions are smaller than patch size")

        N_h, N_w = img_h / patch_size, img_w / patch_size
        if (not N_h.is_integer()) or (not N_w.is_integer()):
            raise Exception("Patch size is not a divisor of image dimensions")

        super().__init__()
        self.patch_size = patch_size
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.d = d
        self.N = int(N_h * N_w)
        self.embed = nn.Conv2d(img_c, d, patch_size, patch_size)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images. Shape: (batch_size, img_c, img_h, img_w)
        :return: Embedded patches of inputs. Shape: (batch_size, N, d)
        """
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class PatchEmbeddings(nn.Module):
    """
    Splits an image into patches, and linearly embeds each of them.
    Options are:
    (2, 44, 2), (12, 24, 12), (14, 20, 14), (18, 12, 18), (19, 10, 19)
    (2, 22, 22, 2), (6, 18, 18, 6), (8, 16, 16, 8), (10, 14, 14, 10), (16, 8, 8, 16), (19, 5, 5, 19), (22, 2, 2, 22)
    (4, 8, 12, 12, 8, 4), (12, 8, 4, 4, 8, 12)
    and any int (or tuple of len 1 containing an int) 48 is divisible by.
    """

    def __init__(self, patch_sizes, d):
        """
        :param patch_sizes: The patch sizes as a tuple
        """
        super().__init__()

        if patch_sizes in patch_dict:
            self.pe = patch_dict[patch_sizes](d)
        elif len(set(patch_sizes)) == 1:
            self.pe = SquarePatchEmbeddings(patch_size=patch_sizes[0], img_h=48, img_w=48, img_c=3, d=d)
        else:
            raise Exception("Given patch sizes are unavailable")

        self.d = d
        self.N = self.pe.N

    def forward(self, x):
        """
        Forward pass
        :param x: Input images. Shape: (batch_size, img_c, img_h, img_w)
        :return: Embedded patches of inputs. Shape: (batch_size, N, d)
        """
        x = self.pe(x)
        return x


class BiasedPatchEmbeddings_2_44_2(nn.Module):
    """
    An embedding class that divides an image to biased, uneven patches
    Assumes image dimensions: (3, 48, 48)
    Patch dimensions are (2, 44, 2)
    """

    def __init__(self, d):
        """
        :param d: The embedding dimension
        """
        super().__init__()
        self.d = d
        self.N = 9
        self.embed = nn.Conv2d(3, d, 44, 44)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images
        :return: Embedded patches of inputs
        """
        bs = x.shape[0]
        x = torch.concat([torch.zeros(bs, 3, 42, 48), x, torch.zeros(bs, 3, 42, 48)], dim=2)
        x = torch.concat([torch.zeros(bs, 3, 132, 42), x, torch.zeros(bs, 3, 132, 42)], dim=3)
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class BiasedPatchEmbeddings_12_24_12(nn.Module):
    """
    An embedding class that divides an image to biased, uneven patches
    Assumes image dimensions: (3, 48, 48)
    Patch dimensions are (12, 24, 12)
    """

    def __init__(self, d):
        """
        :param d: The embedding dimension
        """
        super().__init__()
        self.d = d
        self.N = 9
        self.embed = nn.Conv2d(3, d, 24, 24)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images
        :return: Embedded patches of inputs
        """
        bs = x.shape[0]
        x = torch.concat([torch.zeros(bs, 3, 12, 48), x, torch.zeros(bs, 3, 12, 48)], dim=2)
        x = torch.concat([torch.zeros(bs, 3, 72, 12), x, torch.zeros(bs, 3, 72, 12)], dim=3)
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class BiasedPatchEmbeddings_14_20_14(nn.Module):
    """
    An embedding class that divides an image to biased, uneven patches
    Assumes image dimensions: (3, 48, 48)
    Patch dimensions are (14, 20, 14)
    """

    def __init__(self, d):
        """
        :param d: The embedding dimension
        """
        super().__init__()
        self.d = d
        self.N = 9
        self.embed = nn.Conv2d(3, d, 20, 20)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images
        :return: Embedded patches of inputs
        """
        bs = x.shape[0]
        x = torch.concat([torch.zeros(bs, 3, 6, 48), x, torch.zeros(bs, 3, 6, 48)], dim=2)
        x = torch.concat([torch.zeros(bs, 3, 60, 12), x, torch.zeros(bs, 3, 60, 12)], dim=3)
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class BiasedPatchEmbeddings_18_12_18(nn.Module):
    """
    An embedding class that divides an image to biased, uneven patches
    Assumes image dimensions: (3, 48, 48)
    Patch dimensions are (18, 12, 18)
    """

    def __init__(self, d):
        """
        :param d: The embedding dimension
        """
        super().__init__()
        self.d = d
        self.N = 9
        self.embed = nn.Conv2d(3, d, 18, 18)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images
        :return: Embedded patches of inputs
        """
        bs = x.shape[0]
        # x = torch.concat([x[:, :, :12, :], torch.zeros(bs, 3, 4, 32), x[:, :, 12:, :]], dim=2)
        # x = torch.concat([x[:, :, :, :12], torch.zeros(bs, 3, 36, 4), x[:, :, :, 12:]], dim=3)
        x = torch.concat([x[:, :, :18, :], torch.zeros(bs, 3, 3, 48), x[:, :, 18:30, :], torch.zeros(bs, 3, 3, 48), x[:, :, 30:, :]], dim=2)
        x = torch.concat([x[:, :, :, :18], torch.zeros(bs, 3, 54, 3), x[:, :, :, 18:30], torch.zeros(bs, 3, 54, 3), x[:, :, :, 30:]], dim=3)
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class BiasedPatchEmbeddings_19_10_19(nn.Module):
    """
    An embedding class that divides an image to biased, uneven patches
    Assumes image dimensions: (3, 48, 48)
    Patch dimensions are (19, 10, 19)
    """

    def __init__(self, d):
        """
        :param d: The embedding dimension
        """
        super().__init__()
        self.d = d
        self.N = 9
        self.embed = nn.Conv2d(3, d, 19, 19)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images
        :return: Embedded patches of inputs
        """
        bs = x.shape[0]
        # x = torch.concat([x[:, :, :12, :], torch.zeros(bs, 3, 4, 32), x[:, :, 12:, :]], dim=2)
        # x = torch.concat([x[:, :, :, :12], torch.zeros(bs, 3, 36, 4), x[:, :, :, 12:]], dim=3)
        x = torch.concat([x[:, :, :19, :], torch.zeros(bs, 3, 9, 48), x[:, :, 19:, :]], dim=2)
        x = torch.concat([x[:, :, :, :19], torch.zeros(bs, 3, 57, 9), x[:, :, :, 19:]], dim=3)
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class BiasedPatchEmbeddings_22_4_22(nn.Module):
    """
    An embedding class that divides an image to biased, uneven patches
    Assumes image dimensions: (3, 48, 48)
    Patch dimensions are (22, 4, 22)
    """

    def __init__(self, d):
        """
        :param d: The embedding dimension
        """
        super().__init__()
        self.d = d
        self.N = 9
        self.embed = nn.Conv2d(3, d, 22, 22)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images
        :return: Embedded patches of inputs
        """
        bs = x.shape[0]
        # x = torch.concat([x[:, :, :12, :], torch.zeros(bs, 3, 4, 32), x[:, :, 12:, :]], dim=2)
        # x = torch.concat([x[:, :, :, :12], torch.zeros(bs, 3, 36, 4), x[:, :, :, 12:]], dim=3)
        x = torch.concat([x[:, :, :22, :], torch.zeros(bs, 3, 18, 48), x[:, :, 22:, :]], dim=2)
        x = torch.concat([x[:, :, :, :22], torch.zeros(bs, 3, 66, 18), x[:, :, :, 22:]], dim=3)
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class BiasedPatchEmbeddings_8_16_16_8(nn.Module):
    """
    An embedding class that divides an image to biased, uneven patches
    Assumes image dimensions: (3, 48, 48)
    Patch dimensions are (8, 16, 16, 8)
    """

    def __init__(self, d):
        """
        :param d: The embedding dimension
        """
        super().__init__()
        self.d = d
        self.N = 16
        self.embed = nn.Conv2d(3, d, 16, 16)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images
        :return: Embedded patches of inputs
        """
        bs = x.shape[0]
        x = torch.concat([torch.zeros(bs, 3, 8, 48), x, torch.zeros(bs, 3, 8, 48)], dim=2)
        x = torch.concat([torch.zeros(bs, 3, 64, 8), x, torch.zeros(bs, 3, 64, 8)], dim=3)
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class BiasedPatchEmbeddings_16_8_8_16(nn.Module):
    """
    An embedding class that divides an image to biased, uneven patches
    Assumes image dimensions: (3, 48, 48)
    Patch dimensions are (16, 8, 8, 16)
    """

    def __init__(self, d):
        """
        :param d: The embedding dimension
        """
        super().__init__()
        self.d = d
        self.N = 16
        self.embed = nn.Conv2d(3, d, 16, 16)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images
        :return: Embedded patches of inputs
        """
        bs = x.shape[0]
        x = torch.concat([x[:, :, :24, :], torch.zeros(bs, 3, 16, 48), x[:, :, 24:, :]], dim=2)
        x = torch.concat([x[:, :, :, :24], torch.zeros(bs, 3, 64, 16), x[:, :, :, 24:]], dim=3)
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class BiasedPatchEmbeddings_10_14_14_10(nn.Module):
    """
    An embedding class that divides an image to biased, uneven patches
    Assumes image dimensions: (3, 48, 48)
    Patch dimensions are (10, 14, 14, 10)
    """

    def __init__(self, d):
        """
        :param d: The embedding dimension
        """
        super().__init__()
        self.d = d
        self.N = 16
        self.embed = nn.Conv2d(3, d, 14, 14)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images
        :return: Embedded patches of inputs
        """
        bs = x.shape[0]
        x = torch.concat([torch.zeros(bs, 3, 4, 48), x, torch.zeros(bs, 3, 4, 48)], dim=2)
        x = torch.concat([torch.zeros(bs, 3, 56, 4), x, torch.zeros(bs, 3, 56, 4)], dim=3)
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class BiasedPatchEmbeddings_6_18_18_6(nn.Module):
    """
    An embedding class that divides an image to biased, uneven patches
    Assumes image dimensions: (3, 48, 48)
    Patch dimensions are (6, 18, 18, 6)
    """

    def __init__(self, d):
        """
        :param d: The embedding dimension
        """
        super().__init__()
        self.d = d
        self.N = 16
        self.embed = nn.Conv2d(3, d, 18, 18)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images
        :return: Embedded patches of inputs
        """
        bs = x.shape[0]
        x = torch.concat([torch.zeros(bs, 3, 12, 48), x, torch.zeros(bs, 3, 12, 48)], dim=2)
        x = torch.concat([torch.zeros(bs, 3, 72, 12), x, torch.zeros(bs, 3, 72, 12)], dim=3)
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class BiasedPatchEmbeddings_22_2_2_22(nn.Module):
    """
    An embedding class that divides an image to biased, uneven patches
    Assumes image dimensions: (3, 48, 48)
    Patch dimensions are (22, 2, 2, 22)
    """

    def __init__(self, d):
        """
        :param d: The embedding dimension
        """
        super().__init__()
        self.d = d
        self.N = 16
        self.embed = nn.Conv2d(3, d, 22, 22)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images
        :return: Embedded patches of inputs
        """
        bs = x.shape[0]
        x = torch.concat([x[:, :, :24, :], torch.zeros(bs, 3, 40, 48), x[:, :, 24:, :]], dim=2)
        x = torch.concat([x[:, :, :, :24], torch.zeros(bs, 3, 88, 40), x[:, :, :, 24:]], dim=3)
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class BiasedPatchEmbeddings_19_5_5_19(nn.Module):
    """
    An embedding class that divides an image to biased, uneven patches
    Assumes image dimensions: (3, 48, 48)
    Patch dimensions are (19, 5, 5, 19)
    """

    def __init__(self, d):
        """
        :param d: The embedding dimension
        """
        super().__init__()
        self.d = d
        self.N = 16
        self.embed = nn.Conv2d(3, d, 19, 19)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images
        :return: Embedded patches of inputs
        """
        bs = x.shape[0]
        x = torch.concat([x[:, :, :24, :], torch.zeros(bs, 3, 28, 48), x[:, :, 24:, :]], dim=2)
        x = torch.concat([x[:, :, :, :24], torch.zeros(bs, 3, 76, 28), x[:, :, :, 24:]], dim=3)
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class BiasedPatchEmbeddings_2_22_22_2(nn.Module):
    """
    An embedding class that divides an image to biased, uneven patches
    Assumes image dimensions: (3, 48, 48)
    Patch dimensions are (2, 22, 22, 2)
    """

    def __init__(self, d):
        """
        :param d: The embedding dimension
        """
        super().__init__()
        self.d = d
        self.N = 16
        self.embed = nn.Conv2d(3, d, 22, 22)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images
        :return: Embedded patches of inputs
        """
        bs = x.shape[0]
        x = torch.concat([torch.zeros(bs, 3, 20, 48), x, torch.zeros(bs, 3, 20, 48)], dim=2)
        x = torch.concat([torch.zeros(bs, 3, 88, 20), x, torch.zeros(bs, 3, 88, 20)], dim=3)
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class BiasedPatchEmbeddings_4_8_12_12_8_4(nn.Module):
    """
    An embedding class that divides an image to biased, uneven patches
    Assumes image dimensions: (3, 48, 48)
    Patch dimensions are (4, 8, 12, 12, 8, 4)
    """

    def __init__(self, d):
        """
        :param d: The embedding dimension
        """
        super().__init__()
        self.d = d
        self.N = 36
        self.embed = nn.Conv2d(3, d, 12, 12)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images
        :return: Embedded patches of inputs
        """
        bs = x.shape[0]
        x = torch.concat([x[:, :, :4, :], torch.zeros(bs, 3, 12, 48), x[:, :, 4: 44, :], torch.zeros(bs, 3, 12, 48), x[:, :, 44:, :]], dim=2)
        x = torch.concat([x[:, :, :, :4], torch.zeros(bs, 3, 72, 12), x[:, :, :, 4: 44], torch.zeros(bs, 3, 72, 12), x[:, :, :, 44:]], dim=3)
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class BiasedPatchEmbeddings_12_8_4_4_8_12(nn.Module):
    """
    An embedding class that divides an image to biased, uneven patches
    Assumes image dimensions: (3, 48, 48)
    Patch dimensions are (12_8_4_4_8_12)
    """

    def __init__(self, d):
        """
        :param d: The embedding dimension
        """
        super().__init__()
        self.d = d
        self.N = 36
        self.embed = nn.Conv2d(3, d, 12, 12)

    def forward(self, x):
        """
        Forward pass
        :param x: Input images
        :return: Embedded patches of inputs
        """
        bs = x.shape[0]
        x = torch.concat([x[:, :, :20, :], torch.zeros(bs, 3, 12, 48), x[:, :, 20: 28, :], torch.zeros(bs, 3, 12, 48), x[:, :, 28:, :]], dim=2)
        x = torch.concat([x[:, :, :, :20], torch.zeros(bs, 3, 72, 12), x[:, :, :, 20: 28], torch.zeros(bs, 3, 72, 12), x[:, :, :, 28:]], dim=3)
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


# patch_dict maps tuples of possible biased patch sizes to their appropriate class (for example (12, 8, 4, 4, 8, 12): BiasedPatchEmbeddings_12_8_4_4_8_12)
patch_dict = {tuple(int(val) for val in x): cls[1] for cls in inspect.getmembers(sys.modules[__name__], inspect.isclass) if (x := tuple(cls[0].split('_')[1:])) != ()}
