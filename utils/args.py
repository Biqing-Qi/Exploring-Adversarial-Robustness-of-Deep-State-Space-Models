import argparse
from pathlib import Path

def Build_Parser():
    parser = argparse.ArgumentParser(description='PyTorch Adversarial Training')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N')
    parser.add_argument('--num-classes', type=int, default=10, metavar='N')
    parser.add_argument('--epochs', type=int, default=180, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--workers', type=int, default=12, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--weight-decay', '--wd', default=2e-4,
                        type=float, metavar='W')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--use_AdSS', action='store_true', default=False,
                        help='use AdSS module')
    parser.add_argument('--AdSS_Type', type=str, choices=['relu', 'sgmd', 'tanh'], default='relu',
                        help='the AdSS type used in training')
    parser.add_argument('--epsilon', default=8/255,
                        help='perturbation')
    parser.add_argument('--num-steps', default=10,
                        help='perturb number of steps')
    parser.add_argument('--step-size', default=2/255,
                        help='perturb step size')
    parser.add_argument('--beta', default=6.0,
                        help='regularization, i.e., 1/lambda in TRADES')
    parser.add_argument('--seed', type=int, default=5, metavar='S',
                        help='random seed (default: 5)')
    parser.add_argument('--num_layers', type=int, default=4,
                    help='layers of SSM')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-dir', default='checkpoints',
                        help='directory of model for saving checkpoint')
    parser.add_argument('--AT_type', default='Madry', type=str,
                        help='Madry, TRADE, FreeAT or YOPO in adversrial training stage')
    parser.add_argument('--attack_type', default='PGD', type=str,
                        help='PGD or AA attack in test stage')
    parser.add_argument('--model_name', default='SSM', type=str,
                        help='SSM, DSS, S5, Mega, S6')
    parser.add_argument('--AA_lags', type=int, default=4, metavar='N',
                    help='how many batches to wait before logging training status')
    #! AA evaluation part
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--n_ex', type=int, default=2000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--AA_bs', type=int, default=200)
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--state-path', type=Path, default=None)
    parser.add_argument('--AA_verbose', action='store_true', default=True,
                        help='use the verbose in AA progress')
    return parser