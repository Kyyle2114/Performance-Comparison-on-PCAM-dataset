import torch
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader
import os

from utils.seed import seed_everything

def download_pcam():
    """
    Download PCAM dataset (~ 8.4GB)
    
    type : PIL Image
    location : /code/utils/data/pcam 
    """
    # download pcam dataset 
    torchvision.datasets.PCAM(root='./data', split='train', download=True)
    torchvision.datasets.PCAM(root='./data', split='val', download=True)
    torchvision.datasets.PCAM(root='./data', split='test', download=True)
    
    # remove .gz files 
    path = './data/pcam'
    for file_name in os.listdir(path):
        if file_name.endswith('.gz'):
            file_path = os.path.join(path, file_name)
            os.remove(file_path)
    
    
def load_pcam(path='./data', input_shape=96, augmentation=True, normalize=True, batch_size=256, seed=21):
    """
    Load PCAM dataset (already downloaded)

    Args:
        path (str): path of pcam dataset
        input_shape (int, optional): size of input, (input_shape x input_shape). Defaults to 96.
        augmentation (bool, optional): if True, data augmentation will be applied on train dataloader. Defaults to True.
        normalize (bool, optional): if True, image normaliztion will be applied. Defaults to True.
        batch_size (int, optional): batch size in dataloader. Defaults to 256.
        seed (int, optional): set random seed. Defaults to 21.

    Returns:
        DataLoader: train/val/test DataLoader
    """
    
    # set seed 
    seed_everything(seed)
    
    tf_list = []
    tf_list_test = []
    
    tf_list.append(tr.Resize(input_shape))
    tf_list_test.append(tr.Resize(input_shape))
    
    if augmentation:
        tf_list.append(tr.RandomHorizontalFlip())
        tf_list.append(tr.RandomVerticalFlip())
        tf_list.append(tr.RandomRotation(10))
    
    tf_list.append(tr.ToTensor())
    tf_list_test.append(tr.ToTensor())
    
    if normalize:
        # pre-computed statistic
        tf_list.append(tr.Normalize(mean=[0.70075595, 0.53835785, 0.6916205], 
                                    std=[0.18174392, 0.20083658, 0.16479422]))
        
        tf_list_test.append(tr.Normalize(mean=[0.70075595, 0.53835785, 0.6916205], 
                                         std=[0.18174392, 0.20083658, 0.16479422]))

    transform_train = tr.Compose(tf_list)
    transform_test = tr.Compose(tf_list_test)
        
    trainset = torchvision.datasets.PCAM(root=path, split ='train', download=False, transform=transform_train)
    valset = torchvision.datasets.PCAM(root=path, split ='val', download=False, transform=transform_test)
    testset = torchvision.datasets.PCAM(root=path, split ='test', download=False, transform=transform_test)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader