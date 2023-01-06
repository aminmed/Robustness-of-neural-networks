from __future__ import print_function
import os, os.path
import argparse
from utils import load_project
from tqdm import tqdm
import numpy as np 
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

torch.seed() 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class FGSMAttack(object) : 
    
    def __init__(self, model, eps, min_val = 0., max_val = 255.) -> None:
        
        self.model = model
        self.eps = eps 
        self.min_val = min_val
        self.max_val = max_val
        self.name = "FGSM"

    def FGSM(self, data ,computed_gradients):
        # Collect the element-wise sign of the data gradient
        gradients_sign = computed_gradients.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        new_data = data + self.eps*gradients_sign
        # Adding clipping to maintain [0,255] range
        new_data = torch.clamp(new_data, self.min_val, self.max_val)
        # Return the resulted images 
        return new_data


    def perturb(self, data, target) : 
        
        # Asserting that gradients will be computed for the data tensor ! 
        data.requires_grad = True 
        # pass data through the model to compute gradients and predictions 

        outputs = self.model(data)
        model_predictions = outputs.max(1, keepdim=True)[1]

        #if model_predictions.item() != target.item(): 
        # we use the same loss used to train model 

        criterion = nn.NLLLoss()
        loss = criterion(outputs, target)

        # zero initialize previous gradients 

        self.model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect gradients on the data 
        computed_gradients = data.grad.data
        
        #clean gradients : 
        self.model.zero_grad()
        # Call FGSM method 
        new_data = self.FGSM(data, computed_gradients)

        # Classify new data using initial model 
        new_outputs = self.model(new_data)

        return new_data, new_outputs


def main() : 
    parser = argparse.ArgumentParser()
    parser.add_argument("path_model", metavar="--path-model", nargs="?", default=os.path.join(os.getcwd(), "models/default_model.pth"),
                        help="Path to the project directory to test.")
    parser.add_argument("experiment_dir", metavar="--experiment-dir", nargs="?", default=os.path.join(os.getcwd(), "experiments/FGSM"),
                        help="Path to save experiment.")
    
    parser.add_argument("-b", "--batch-size", type=int, default=256,
                        help="Set batch size.")
    parser.add_argument("-e", "--epsilon", type=float, default=4/255,
                        help="Set epsilon (noise parameter).")

    args = parser.parse_args()
    batch_size = args.batch_size
    experiment_directory = os.path.join(args.experiment_dir,\
                                        'FGSM_eps_' + str(args.epsilon * 255.)[:2]+\
                                        '_time_' + time.strftime("%Y%m%d%H%M%S") ) 
    os.mkdir(experiment_directory)
    print("Load default model ...")
    project_module = load_project(os.getcwd())
    net = project_module.Net(project_module.ResidualBlock, [2, 2, 2])
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
    print("Start FGSM attack ...")
    FGSM_model = FGSMAttack(model = net, eps = args.epsilon)

    adverserial_images = []
    y_preds = []
    y_preds_adv = []
    adv_confidences = []
    init_confidences = []
    labels_np = []
    total = 0
    correct = 0
    adv_correct = 0
    misclassified = 0

    for i, (data, targets) in tqdm(enumerate(te_loader)) : 
        data, targets = data.to(device), targets.to(device)

        init_outputs = net(data)
        init_confidence, init_predictions = init_outputs.max(1, keepdim = False)

        adv_data, adv_outputs = FGSM_model.perturb(data, targets)
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
