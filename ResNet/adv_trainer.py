import argparse
import os
from utils import create_logger, argument_parser, load_project
from tqdm import tqdm 
from time import time 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv

from FGSMAttack import FGSMAttack
from L2PGDAttack import L2PGDAttack
from LinfPGDAttack import LinfPGDAttack

torch.seed() 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



class AdvTrainer(object): 
    
    def __init__(self, attack,logger, args) -> None:
        
        self.attack = attack
        self.args = args 
        self.logger = logger 
        self.criterion = nn.NLLLoss()
        self.train_std_acc_tracker = []
        self.train_adv_acc_tracker = []
        self.val_std_acc_tracker = []
        self.val_adv_acc_tracker = []


    
    def train(self, model, train_loader, val_loader) : 
        
        args = self.args
        logger = self.logger

        opt = torch.optim.SGD(model.parameters(), args.learning_rate, 
                              weight_decay=args.weight_decay,
                              momentum=args.momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, 
                                                         milestones=[40000, 60000], 
                                                         gamma=0.1)


        _iter = 0 # iteration of SGD 

        for epoch in range(1, args.max_epoch) :
            epoch_start = time()
            with tqdm(train_loader, unit="batch") as tepoch:
                for data, targets in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    data, targets = data.to(device), targets.to(device)
                    # create adversarial examples from original data 
                    adv_data, _ = self.attack.perturb(data, targets)
                    adv_data = adv_data.to(device)
                    # pass the adversarial examples through the model 
                    adv_outputs = model(adv_data)           
                    loss = self.criterion(adv_outputs, targets)
                    # backpropagate loss gradients ... 
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    # evaluation during training; we separte adv samples
                    # predictions evaluation from original images 
                    # predictions evaluation

                    if _iter % args.n_eval_step == 0 : 
                        t1 = time()
                        with torch.no_grad() : 
                            org_outputs = model(data)                # evaluate on original data 
                        org_preds = torch.max(org_outputs, dim=1)[1]
                        org_acc = (org_preds == targets).sum().item()
                        org_acc = (org_acc * 100) / targets.size(0)
                        # evaluate on adv data 
                        adv_preds = torch.max(adv_outputs, dim=1)[1]
                        adv_acc = (adv_preds == targets).sum().item()
                        adv_acc = (adv_acc * 100) / targets.size(0)
                        t2 = time()
                        self.train_std_acc_tracker.append(org_acc)
                        self.train_adv_acc_tracker.append(adv_acc)

                        logger.info(f'Epoch {epoch}: iter {_iter}, lr={opt.param_groups[0]["lr"]}, '+\
                                    f'standard acc: {org_acc:.3f}%,adv acc: {adv_acc:.3f}%'+\
                                    f'spent {time()-epoch_start:.2f}s,tr_loss: {loss.item():.3f}')
                    
                    if _iter % args.n_checkpoint_step == 0:
                        file_name = os.path.join(args.model_folder, f'checkpoint_'
                                                    + self.attack.name + f'_{_iter}.pth')
                        model.save(file_name)

                    _iter += 1
                    # scheduler depends on training interation
                    scheduler.step()
                if val_loader is not None:
                    
                    t1 = time()
                    va_acc, va_adv_acc = self.test(model, val_loader, True)
                    va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0
                    t2 = time()
                    self.val_std_acc_tracker.append(va_acc)
                    self.val_adv_acc_tracker.append(va_adv_acc)
                        
                    logger.info('\n'+'='*20 +f' evaluation at epoch: {epoch} iteration: {_iter} ' \
                    +'='*20)
                    logger.info(f'test acc: {va_acc:.3f}%, test adv acc: {va_adv_acc:.3f}%, spent: {t2-t1:.3f} s')
                    logger.info('='*28+' end of evaluation '+'='*28+'\n')


    def test(self, model, ts_loader, adv_test = False) : 

        total_org_acc = 0.0
        num = 0
        total_adv_acc = 0.0

        with torch.no_grad() : 

            with tqdm(ts_loader, unit="batch") as tepoch:
                
                for data, targets in tepoch:
                    tepoch.set_description("Test : ")
                    data, targets = data.to(device), targets.to(device)
                    org_output = model(data)
                    org_preds = torch.max(org_output, dim=1)[1]
                    org_acc = (org_preds == targets).sum().item()
                    org_acc = (org_acc * 100) / targets.size(0)
                    
                    adv_acc = 0 

                    if adv_test : 
                        with torch.enable_grad():  # enable grad to compute the sign of gradients ! 
                            adv_data, _ = self.attack.perturb(data, targets)
                        # pass the adversarial examples through the model 
                        adv_outputs = model(adv_data)
                        adv_preds = torch.max(adv_outputs, dim=1)[1]
                        adv_acc = (adv_preds == targets).sum().item()
                        adv_acc = (adv_acc * 100) / targets.size(0)

                    total_org_acc += org_acc
                    total_adv_acc += adv_acc
                    num += targets.size(0)

        return total_org_acc / num, total_adv_acc / num 
            


def main() : 

    args = argument_parser()

    save_folder = '%s_%s' % ('CIFAR10', args.attack_type)

    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    if not os.path.exists(model_folder) : 
        os.mkdir(model_folder)

    logger = create_logger(log_folder, args.todo, 'info')
    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    print("Load CNN model ...")
    project_module = load_project(os.getcwd())
    net = project_module.Net()
    net.load_for_testing(project_dir='.')
    net.to(device)


    if args.attack_type == 'FGSM' : 
        attack = FGSMAttack(model = net, eps = args.epsilon)
    elif  args.attack_type == 'L2PGD' : 
        attack = L2PGDAttack(model = net, epsilon = args.epsilon, alpha=8/255,iteration=40)
    else : 
        attack = LinfPGDAttack(model = net, epsilon = args.epsilon, alpha=8/255,iteration=40)

    trainer = AdvTrainer(attack, logger, args)

    if args.todo == 'train':
        transform_train = tv.transforms.Compose([
                tv.transforms.ToTensor(),
            ])
        tr_dataset = tv.datasets.CIFAR10(args.data_root, 
                                       train=True, 
                                       transform=transform_train, 
                                       download=True)

        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # evaluation during training
        te_dataset = tv.datasets.CIFAR10(args.data_root, 
                                       train=False, 
                                       transform=tv.transforms.ToTensor(), 
                                       download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        trainer.train(net, tr_loader, te_loader)
        np.save('train_adv_acc_tracker.npy',np.array(trainer.train_adv_acc_tracker))
        np.save('train_std_acc_tracker.npy',np.array(trainer.train_std_acc_tracker))
        np.save('val_adv_acc_tracker.npy',np.array(trainer.val_adv_acc_tracker))
        np.save('val_std_acc_tracker.npy',np.array(trainer.val_std_acc_tracker))
    elif args.todo == 'test':
        
        te_dataset = tv.datasets.CIFAR10(args.data_root, 
                                       train=False, 
                                       transform=tv.transforms.ToTensor(), 
                                       download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        checkpoint = torch.load(args.load_checkpoint)

        #net.load(checkpoint)

        org_acc, adv_acc = trainer.test(net, te_loader, adv_test=True)

        print(f"orginal images acc: {org_acc * 100:.3f}%, adversarial images acc: {adv_acc * 100:.3f}%")

    else:
        raise NotImplementedError



if __name__ == '__main__' : 
    main() 



