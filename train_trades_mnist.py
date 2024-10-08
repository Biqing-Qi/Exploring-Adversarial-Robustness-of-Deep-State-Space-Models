import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from autoattack import AutoAttack
from utils.args import Build_Parser
from utils.evalution import eval_test, adv_test, adv_eval_train, eval_train
from utils.mnist import build_model
from utils.trades import train

# args
parser = Build_Parser()
args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.ToTensor()),
                   batch_size=args.test_batch_size, shuffle=False, **kwargs)

def AA_adv_test(model,args,log_path):
    model.eval()
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=log_path,
        version=args.version,verbose=args.AA_verbose)
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    torch.cuda.empty_cache()
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                bs=args.AA_bs, state_path=args.state_path)

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                y_test[:args.n_ex], bs=args.batch_size)

def main():
    if args.use_AdSS:
        log_path = os.path.join(model_dir,'train_log'+args.model_name+'AdSS_{}'.format(args.AdSS_Type)+args.AT_type+'.txt')
    else:    
        log_path = os.path.join(model_dir,'train_log'+args.model_name+args.AT_type+'.txt')
    log_file = open(log_path, 'w')
    model = build_model(args, args.model_name).to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    for epoch in range(1, args.epochs + 1):
        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch, args.AT_type)

        print('================================================================')
        if args.AT_type != 'Nat' and epoch % args.AA_lags == 0:
            with open(log_path, 'a') as f:
                f.write('now epoch:' + str(epoch) + '\n')
                f.flush()
            AA_adv_test(model, args, log_path)
        # save checkpoint
        if args.use_AdSS:
            torch.save(model.state_dict(),
                    os.path.join(model_dir, args.model_name+args.AT_type+'{}'.format(args.AdSS_Type)+ '-epoch{}.pt'.format(epoch)))
        else:
            torch.save(model.state_dict(),
                    os.path.join(model_dir, args.model_name+args.AT_type+ '-epoch{}.pt'.format(epoch)))
        scheduler.step()


if __name__ == '__main__':
    main()
