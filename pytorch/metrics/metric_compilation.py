import torch
import torch.nn as nn
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import wandb

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def effective_rank(Z):
    svd = torch.linalg.svdvals(Z)

    rankme_score = 0
    svd_norm = torch.norm(svd, p=1, dim=-1)
    eppsilon = 1e-7
    for i in range(0, len(svd)):
        p_k = svd[i] / svd_norm + eppsilon
        rankme_score = rankme_score + p_k * torch.log(p_k)
    return torch.exp(-1 * rankme_score)


def visualize_spectrum(Z):
    svd = torch.log(torch.linalg.svdvals(Z)).detach().cpu().numpy()
    sns.set()
    plt.figure(figsize=(6,4))
    sns.lineplot(x=list(range(len(svd))), y=svd)
    plt.xlabel('Index')
    plt.ylabel('Log Singular Value')
    plt.title('Singular Value Spectrum')
    wandb.log({"sns_plot": wandb.Image(plt)})


def visualize_umap(Z):
    pass

def knn_accuracy(Z, labels, k=5):

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Z.cpu().numpy(), labels.cpu().numpy())
    preds = knn.predict(Z.cpu().numpy())
    acc = accuracy_score(labels.cpu().numpy(), preds)
    return acc


    