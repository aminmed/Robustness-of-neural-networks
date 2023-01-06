import os, os.path, sys
import argparse

import importlib.abc
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import load_project
import time
import numpy as np

## Code structure inspired from https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR 
## In this Repo they implement an advanced PGD, We have changed it to have the classical PGD
## as a first version
'''
   Hyperparameters:
   epsilon: 
   alpha : 
   itarations:
'''
torch.seed() 
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class L2PGDAttack(object):
    def __init__(self, model, epsilon = 0.025,alpha=2/255,iteration=40):
        self.model = model
        self.epsilon=epsilon
        self.alpha=alpha
        self.iteration=iteration
        self.name = "L2PGD"

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        ori_images = x.data
        for i in range(self.iteration):
            x.requires_grad_()
            with torch.enable_grad():
                outputs = self.model(x)
                loss = F.cross_entropy(outputs, y)
            loss.backward()
            adv_images = x.detach() + self.alpha * torch.sign(x)
            eta = torch.clamp(adv_images - ori_images, min=-self.epsilon, max=self.epsilon)
            x = torch.clamp(ori_images + eta, min=0., max=255.).detach_()
        return x , self.model(x)
        
def main() : 
    parser = argparse.ArgumentParser()
    parser.add_argument("path_model", metavar="path-model", nargs="?", default=os.path.join(os.getcwd(), "models/default_model.pth"),
                        help="Path to the project directory to test.")
    parser.add_argument("experiment_dir", metavar="experiment-dir", nargs="?", default=os.path.join(os.getcwd(), "experiments/L2PGD"),
                        help="Path to save experiment.")
    
    parser.add_argument("-b", "--batch-size", type=int, default=256,
                        help="Set batch size.")
    parser.add_argument("-e", "--epsilon", type=float, default=0.25,
                        help="Set epsilon (noise parameter).")
    args = parser.parse_args()
    batch_size = args.batch_size
    experiment_directory = os.path.join(args.experiment_dir,\
                                        'L2PGD_eps_' + str(args.epsilon * 255.)[:4]+\
                                        '_time_' + time.strftime("%Y%m%d%H%M%S") ) 
    os.mkdir(experiment_directory)
    print("Load default model ...")
    project_module = load_project(os.getcwd())
    net = project_module.Net()
    net.to(device)
    net.load(args.path_model)
    print("Default model loaded ...")
    print("Load CIFAR10 test data ...")
    te_dataset = tv.datasets.CIFAR10("./data", 
                                       train=False, 
                                       transform=transforms.Compose([transforms.ToTensor()]), 
                                       download=True)
    te_loader = DataLoader( te_dataset,
                            batch_size=batch_size, 
                            shuffle=False, num_workers=4)
    print("CIFAR10 test dataset loaded ...")
    print("Start L2PGD attack ...")
    L2PGD_model = L2PGDAttack(model = net, epsilon = args.epsilon,alpha=2/255,iteration=40)
    adverserial_images = []
    y_preds = []
    y_preds_adv = []
    adv_confidences = []
    init_confidences = []
    total = 0
    correct = 0
    adv_correct = 0
    labels_np = []
    misclassified = 0

    for i, (data, targets) in tqdm(enumerate(te_loader)) : 
         data, targets = data.to(device), targets.to(device)

         init_outputs = net(data)
         init_confidence, init_predictions = init_outputs.max(1, keepdim = False)

         adv_data , adv_outputs = L2PGD_model.perturb(data, targets)

         adv_confidence, adv_predictions = adv_outputs.max(1, keepdim = False) 

         total += targets.size(0)
         correct += (init_predictions == targets).sum().item()
         adv_correct += (adv_predictions == targets).sum().item()
         misclassified += (init_predictions != adv_predictions).sum().item()  
         adverserial_images.extend((adv_data).cpu().data.numpy())
         y_preds.extend(init_predictions.cpu().data.numpy())
         y_preds_adv.extend(adv_predictions.cpu().data.numpy())
         adv_confidences.extend(adv_confidence.cpu().data.numpy())
         init_confidences.extend(init_confidence.cpu().data.numpy())
         labels_np.extend(targets.cpu().data.numpy())
    np.save(experiment_directory + '/adverserial_images.npy',adverserial_images)  
    np.save(experiment_directory + '/y_GT.npy',labels_np)  
    np.save(experiment_directory + '/y_preds.npy',y_preds)
    np.save(experiment_directory + '/y_preds_adv.npy',y_preds_adv)
    np.save(experiment_directory + '/adv_confidences.npy',adv_confidences)
    np.save(experiment_directory + '/init_confidences.npy',init_confidences)
    
    print('Accuracy of the model w/0 adverserial attack on test images is : {} %'.format(100*correct/total))
    print('Accuracy of the model with adverserial attack on test images is : {} %'.format(100* adv_correct/total))
    print('Number of misclassified examples: {}/{}, success precentage : {}%'.format(misclassified,total, misclassified*100/total ))
                        
if __name__ == '__main__' : 
    main()
