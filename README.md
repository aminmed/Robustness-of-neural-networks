# Bayesian adversarial training to improve the robustness of Machine Learning models against white-box attacks
Here is an implementation of adversarial training for image classification which is designed to defend against white box attacks, specifically FGSM, L2 PGD, and Linf PGD. It is built using the CIFAR dataset and includes the following model architectures:

- The initial model provided in the lecture.

- Residual Networks.

- VGGs

- Byesian Neural Networks

Our implementation is based on python and it's inspired from this articles:

- Towards Deep Learning Models Resistant to Adversarial Attacks: https://arxiv.org/pdf/1706.06083.pdf

- Robustness of Bayesian Neural Networks to White-Box Adversarial Attacks: https://arxiv.org/pdf/2111.08591.pdf

- Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network: https://arxiv.org/abs/1810.01279


and these githubs repositories: 

- https://github.com/xuanqing94/BayesianDefense


- https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR


## Dependencies:
 ````
 numpy, pandas, pytorch, torchvision
 ```` 
 
## How to train ?
#### Train vanilla VGG16 with adversarial training 
	
 	cd ./BasicModel 
 	chmod +x adv_train.sh 
	sh ./adv_train.sh
	
#### Train BNN VGG16 and BNN VGG16 with bayesian adversarial training
	
 	cd ./Bayesian_adversarial_training 
 	chmod +x train_BNN_adv.sh 
	sh ./train_BNN_adv.sh
	

## How to test ?
#### Test trained models using LinfPGD
	
 	cd ./BasicModel 
 	chmod +x linfpgd.sh 
	sh ./linfpgd.sh

#### Test trained models using L2PGD
	
 	cd ./BasicModel 
 	chmod +x l2pgd.sh 
	sh ./l2pgd.sh
	
#### Test trained models using FGSM
	
 	cd ./BasicModel 
 	chmod +x fgsm.sh 
	sh ./fgsm.sh	

## Contributors :

  - BENCHEIKH LEHOCINE Mohammed Amine
  
  - DJECTA Hibat_Errahmen
  
  - KHEDIM Ibtissem
  
