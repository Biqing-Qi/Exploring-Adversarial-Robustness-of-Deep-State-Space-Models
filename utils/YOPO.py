import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn

class FastGradientLayerOneTrainer(object):

    def __init__(self, Hamiltonian_func, param_optimizer,
                    inner_steps=2, sigma = 0.008, eps = 0.03):
        self.inner_steps = inner_steps
        self.sigma = sigma
        self.eps = eps
        self.Hamiltonian_func = Hamiltonian_func
        self.param_optimizer = param_optimizer

    def step(self, inp, p, eta):
        '''
        Perform Iterative Sign Gradient on eta
        ret: inp + eta
        '''

        p = p.detach()

        for i in range(self.inner_steps):
            tmp_inp = inp + eta
            tmp_inp = torch.clamp(tmp_inp, 0, 1)
            H = self.Hamiltonian_func(tmp_inp, p)

            eta_grad = torch.autograd.grad(H, eta, only_inputs=True, retain_graph=False)[0]
            eta_grad_sign = eta_grad.sign()
            eta = eta - eta_grad_sign * self.sigma

            eta = torch.clamp(eta, -1.0 * self.eps, self.eps)
            eta = torch.clamp(inp + eta, 0.0, 1.0) - inp
            eta = eta.detach()
            eta.requires_grad_()
            eta.retain_grad()

        yofo_inp = eta + inp
        yofo_inp = torch.clamp(yofo_inp, 0, 1)
        loss = -1.0 * (self.Hamiltonian_func(yofo_inp, p) -
                       5e-4 * cal_l2_norm(self.Hamiltonian_func.layer))

        loss.backward()

        return yofo_inp, eta


class Hamiltonian(_Loss):

    def __init__(self, layer, reg_cof = 1e-4):
        super(Hamiltonian, self).__init__()
        self.layer = layer
        self.reg_cof = 0


    def forward(self, x, p):

        B,C,H,W = x.shape
        x = x.view(B, H*W, C)
        y = self.layer(x)

        bs, lenth, dim = y.shape
        p = p.view(bs, lenth, dim)
        H = torch.sum(y * p)
        return H



class CrossEntropyWithWeightPenlty(_Loss):
    def __init__(self, module, DEVICE, reg_cof = 1e-4):
        super(CrossEntropyWithWeightPenlty, self).__init__()

        self.reg_cof = reg_cof
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)
        self.module = module


    def __call__(self, pred, label):
        cross_loss = self.criterion(pred, label)
        weight_loss = 0

        weight_loss = cal_l2_norm(self.module)

        loss = cross_loss + self.reg_cof * weight_loss
        return loss

def cal_l2_norm(layer: torch.nn.Module):
    loss = 0.
    for name, param in layer.named_parameters():
        if name == 'weight':
            loss = loss + 0.5 * torch.norm(param,) ** 2

    return loss

def adv_train(args, model, device, train_loader, optimizer, epoch, LayerOneTrainer):
    criterion = nn.CrossEntropyLoss()
    # switch to train mode
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)

        eta = torch.FloatTensor(*data.shape).uniform_(args.epsilon, args.epsilon)
        eta = eta.to(label.device)
        eta.requires_grad_()

        optimizer.zero_grad()
        LayerOneTrainer.param_optimizer.zero_grad()

        for j in range(5):
            #optimizer.zero_grad()

            TotalLoss = 0

            pred = model(data + eta.detach())

            loss = criterion(pred, label)
            TotalLoss = TotalLoss + loss
            TotalLoss.backward()
            p = -1.0 * model.input_out.grad
            yofo_inp, eta = LayerOneTrainer.step(data, p, eta)


        optimizer.step()
        LayerOneTrainer.param_optimizer.step()
        optimizer.zero_grad()
        LayerOneTrainer.param_optimizer.zero_grad()
        if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))