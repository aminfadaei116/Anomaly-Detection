import cv2
from scipy.io import loadmat
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
     


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


