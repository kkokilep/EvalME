import yaml
import torch
from torchvision import datasets, transforms
import wandb
#### Dataset Imports####
from .datasets.Cifar100 import return_Cifar100_dataloader, return_Cifar100_normalization
from .datasets.Cifar10 import return_Cifar10_dataloader, return_Cifar10_normalization
#### Model Imports####
from .model_implementation.resnet import return_resnet18, return_resnet50

#### Metric Imports####
from .metrics.metric_compilation import effective_rank, visualize_spectrum, knn_accuracy

from tqdm import tqdm
def get_dataloader(cfg):
    if cfg['dataset']['name'] == 'cifar100':
        transform_list = get_augmentations(cfg)
        transform_list.append(return_Cifar100_normalization())
        
        testloader = return_Cifar100_dataloader(batch_size=cfg['dataset']['batch_size'], transform=transforms.Compose(transform_list), 
                                                num_workers=cfg['dataset']['num_workers'],root=cfg['dataset']['root'])
    elif cfg['dataset']['name'] == 'cifar10':
        transform_list = get_augmentations(cfg)
        transform_list.append(return_Cifar10_normalization())
        
        testloader = return_Cifar10_dataloader(batch_size=cfg['dataset']['batch_size'], transform=transforms.Compose(transform_list), 
                                                num_workers=cfg['dataset']['num_workers'],root=cfg['dataset']['root'])
    
    
    return testloader

def get_augmentations(cfg):
    augmentations = []
    if cfg['augmentation']['random_crop']:
        augmentations.append(transforms.RandomCrop(cfg['augmentation']['size'], padding=4))
    if cfg['augmentation']['resize']:
        augmentations.append(transforms.Resize(cfg['dataset']['img_size']))
    if cfg['augmentation']['random_horizontal_flip']:
        augmentations.append(transforms.RandomHorizontalFlip())
    if cfg['augmentation']['random_rotation']:
        augmentations.append(transforms.RandomRotation(cfg['augmentation']['random_rotation']))
    if cfg['augmentation']['color_jitter']:
        augmentations.append(transforms.ColorJitter())
    
    augmentations.append(transforms.ToTensor())
    return augmentations

def get_model(cfg):
    if cfg['model']['name'] == 'resnet18':
        model = return_resnet18(cfg)
    elif cfg['model']['name'] == 'resnet50':
        model = return_resnet50(cfg)


    # Note that this is for checkpoints taken from solo-learn library. Need to make it more general for other checkpoints.
    state_dict = torch.load(cfg['model']['ckpt'])
    state = state_dict['state_dict']

    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder", "backbone")] = state[k]
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    model.load_state_dict(state)
    model = model.to(cfg['experiment']['device'])
    return model

def evaluate(cfg):
    test_loader = get_dataloader(cfg)
    model = get_model(cfg)
    model.eval()

    # Note that this is for classification tasks. Need to make it more general for other tasks.
    output_matrix = []
    label_matrix = []
    for i, (data,labels) in tqdm(enumerate(test_loader)):
        data, labels = data.to(cfg['experiment']['device']), labels.to(cfg['experiment']['device'])
        with torch.no_grad():
            outputs = model(data)
        output_matrix.append(outputs.detach().cpu())
        label_matrix.append(labels.detach().cpu())
        
    output_matrix = torch.cat(output_matrix)
    label_matrix = torch.cat(label_matrix)



    ############################# Metrics to Analyze Representations  ##############################
    if cfg['metrics']['effective_rank']:
        effective_rank_val = effective_rank(output_matrix)
        wandb.log({"test/effective_rank": effective_rank_val})
    
    if cfg['metrics']['visualize_spectrum']:
        visualize_spectrum(output_matrix)
    
    if cfg['metrics']['knn_acuracy']:
        knn_acc = knn_accuracy(output_matrix, label_matrix)
        wandb.log({"test/knn_accuracy": knn_acc})
        