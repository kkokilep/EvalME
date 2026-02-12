from torchvision import datasets, transforms
import torch

def return_Cifar100_dataloader(batch_size=128, transform=None, num_workers=2,root='/data/'):
    testset = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return testloader

def return_Cifar100_normalization():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])