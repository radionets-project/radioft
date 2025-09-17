import torch


def isdouble(dtype):
    return dtype is (torch.double or torch.float64)
