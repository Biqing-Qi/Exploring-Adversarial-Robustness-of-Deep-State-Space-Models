import torch
import torch.nn.functional as F
from utils.used_attacks import PGD
from autoattack.autopgd_base import APGDAttack as APGD

def eval_train(args, model, device, train_loader):
    model.eval()
    
    train_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
                
    train_loss /= len(train_loader.dataset)

    print('Train: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    train_loss, correct, len(train_loader.dataset),
    100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    
    return train_loss, training_accuracy

def adv_eval_train(args, model, device, train_loader, attack_method = 'PGD'):
    model.eval()
    
    train_loss = 0
    correct = 0
    
    if attack_method == 'PGD':
        attack = PGD(model, args.epsilon, args.step_size, args.num_steps, normalize=False)
    elif args.attack_type =='APGD':
        attack = APGD(model, eps=args.epsilon, seed=args.seed,device=device)

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        if args.attack_type =='APGD':
            data_adv = attack.perturb(data, target)
        else:
            data_adv = attack(data, target)
        with torch.no_grad():
            output = model(data_adv)
                    
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
                
    train_loss /= len(train_loader.dataset)

    print('Adv Train: Average loss: {:.4f}, Adv Accuracy: {}/{} ({:.0f}%)'.format(
    train_loss, correct, len(train_loader.dataset),
    100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    
    return train_loss, training_accuracy

def adv_test(args, model, device, test_loader, attack_method = 'PGD'):
    model.eval()

    test_loss = 0
    correct = 0
    if attack_method == 'PGD':
        attack = PGD(model, args.epsilon, args.step_size, args.num_steps, normalize=False)
    elif args.attack_type =='APGD':
        attack = APGD(model, eps=args.epsilon, seed=args.seed,device=device)
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        if args.attack_type =='APGD':
            data_adv = attack.perturb(data, target)
        else:
            data_adv = attack(data, target)
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            output = model(data_adv)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            torch.cuda.empty_cache()
            
    test_loss /= len(test_loader.dataset)

    print('Adv Test: Average loss: {:.4f}, Adv Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
                
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        torch.cuda.empty_cache()
           
    test_loss /= len(test_loader.dataset)

    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy
    