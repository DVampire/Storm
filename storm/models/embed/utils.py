import torch
from timm.models.layers import to_2tuple

def get_patch_info(sample_size, patch_size):
    """
    sample: (N, C, T, S, F)
    x: (N, L, patch_size[0] * patch_size[2] * patch_size[2] * C)
    """
    N, C, T, S, F = sample_size

    p1 = patch_size[0]
    p2 = patch_size[1]
    p3 = patch_size[2]

    assert T % p1 == 0 and S % p2 == 0 and F % p3 == 0

    n1 = T // p1
    n2 = S // p2
    n3 = F // p3

    patch_info = (N, C, T, S, F, p1, p2, p3, n1, n2, n3)

    return patch_info


def patchify(sample, patch_info = None):
    """
    sample: (N, C, T, S, F)
    return: (N, L, patch_size[0] * patch_size[2] * patch_size[2] * C)
    """

    N, C, T, S, F, p1, p2, p3, n1, n2, n3 = patch_info

    x = sample.reshape(shape=(N, C, n1, p1, n2, p2, n3, p3))
    x = torch.einsum("nctuhpwq->nthwupqc", x)
    x = x.reshape(shape=(N, n1 * n2 * n3, p1 * p2 * p3 * C))

    return x


def unpatchify(sample, patch_info):
    """
    sample: (N, L, patch_size[0] * patch_size[2] * patch_size[2] * C)
    return: (N, C, T, S, F)
    """

    N, C, T, S, F, p1, p2, p3, n1, n2, n3 = patch_info

    sample = sample.reshape(shape=(N, n1, n2, n3, p1, p2, p3, C))
    sample = torch.einsum("nthwupqc->nctuhpwq", sample)
    sample = sample.reshape(shape=(N, C, T, S, F))

    return sample