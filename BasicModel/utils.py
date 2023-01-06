import argparse
import importlib
import importlib.abc
import logging 
import os
import sys 

def create_logger(save_path='', file_type='', level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        
        fh.setLevel(_level)

        logger.addHandler(fh)
        logger.propagate = False
    return logger

def argument_parser() : 
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("model_root", metavar="model-root", nargs="?", default=os.path.join(os.getcwd(), "models"),
                        help="Path to root folder where to save checkpoints of the model.")
    
    parser.add_argument("data_root", metavar="data-root", nargs="?", default=os.path.join(os.getcwd(), "data"),
                        help="Path where to find the dataset.")

    parser.add_argument("log_root", metavar="log-root", nargs="?", default=os.path.join(os.getcwd(), "logs"),
                        help="Path to root folder where where to save logs files training.")

    parser.add_argument('--load_checkpoint', default='./model/default_model.pth')
    parser.add_argument('--todo', choices=['train', 'test'], default='train')

    # parameters for generating adversarial examples
    parser.add_argument('--epsilon', '-e', type=float, default=0.0157, 
        help='maximum perturbation of adversaries (4/255=0.0157)')

    # parameters for optimizer and training  
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
    parser.add_argument('--max_epoch', '-m_e', type=int, default=200, 
        help='the maximum numbers of the model see a sample')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='momentum for optimizer')
    parser.add_argument('--weight_decay', '-w', type=float, default=2e-4, 
        help='the parameter of l2 restriction for weights')


    parser.add_argument('--n_eval_step', type=int, default=200, 
        help='number of iteration per one evaluation')
    parser.add_argument('--n_checkpoint_step', type=int, default=800, 
        help='number of iteration to save a checkpoint')   

    parser.add_argument("--attack-type", choices=['FGSM', 'L2PGD', 'LinfPGD'], default='FGSM')
    
    args = parser.parse_args()

    return args 

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))

def load_project(project_dir):
    module_filename = os.path.join(project_dir, 'model.py')
    if os.path.exists(project_dir) and os.path.isdir(project_dir) and os.path.isfile(module_filename):
        print("Found valid project in '{}'.".format(project_dir))
    else:
        print("Fatal: '{}' is not a valid project directory.".format(project_dir))
        raise FileNotFoundError 

    sys.path = [project_dir] + sys.path
    spec = importlib.util.spec_from_file_location("model", module_filename)
    project_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_module)

    return project_module