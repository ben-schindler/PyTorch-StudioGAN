import torch
import torch.nn as nn
from discr_ensemble import DiscriminatorEnsemble

from src.models.deep_conv import Generator as Gen
from src.models.deep_conv import Discriminator as Discr

def Generator(*args, **kwargs):
    return Gen(*args, **kwargs)

def Discriminator(multiplier=1, *args, **kwargs):
    return DiscriminatorEnsemble(discr_class=Discr, config=None, multiplier=multiplier, weighting='ew', *args, **kwargs)