import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import random


class GMS:
    def __init__(self, optimizer, reduction='mean', writer=None):
        self._optim, self._reduction = optimizer, reduction

    @property
    def optimizer(self):
        return self._optim
    
    def state_dict(self):
        return self._optim.state_dict()

    def load_state_dict(self, state_dict):
        self._optim.load_state_dict(state_dict)

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives, ddp_model=None):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''
        param_idxs = []
        idx = -1
        for name, _ in ddp_model.named_parameters():
            idx += 1
            if 'q_emb' in name or 'q_net' in name:
                param_idxs.append(idx)

        grads, shapes, has_grads, other_grads = self._pack_grad(objectives, ddp_model, param_idxs)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad, param_idxs, other_grads)
        return

    def _project_conflicting(self, grads, has_grads):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        coefs = torch.ones(num_task, dtype=torch.float32, device=grads[0].device)
        cnt = 0
        for i in range(len(pc_grad)):
            g_i = pc_grad[i]
            indices = list(range(num_task))
            random.shuffle(list(range(num_task)))
            random.shuffle(grads)
            for index in indices:
                g_j = grads[index]
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    coef = g_i_g_j / (g_j.norm() ** 2)
                    cnt += 1
                    g_i -= coef * g_j
                    coefs[index] -= coef
        
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).sum(dim=0)
        else:
            exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        # print('cnt:', cnt)
        return merged_grad

    def _set_grad(self, grads, param_idxs, other_grads):
        '''
        set the modified gradients to the network
        '''

        i, j = 0, 0
        for group in self._optim.param_groups:
            for p in group['params']:
                if (i + j) not in param_idxs:
                    p.grad = other_grads[j]
                    j += 1
                else:
                    p.grad = grads[i]
                    i += 1
        return

    def _pack_grad(self, objectives, ddp, param_idxs):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads, other_grads = [], [], [], []
        for ii, obj in enumerate(objectives):
            self._optim.zero_grad(set_to_none=True)
            # if ii == 0: continue
            # out_tensors = list(_find_tensors(obj))
            # ddp.reducer.prepare_for_backward(out_tensors)
            if ii < len(objectives) - 1:
                obj.backward(retain_graph=True)
                last = False
                grad, shape, has_grad = self._retrieve_grad(param_idxs, last)
            else:
                obj.backward(retain_graph=False)
                last = True
                grad, shape, has_grad, other_grad = self._retrieve_grad(param_idxs, last)
                other_grads = other_grad
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads, other_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = int(np.prod(shape))
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self, param_idxs, last = False):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad, other_grad = [], [], [], []
        idx = -1
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                idx += 1
                if last and idx not in param_idxs:
                    if p.grad is None:
                        other_grad.append(torch.zeros_like(p).to(p.device))
                        continue
                    other_grad.append(p.grad.clone())
                    continue
                if idx not in param_idxs:
                    continue
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        if last:
            return grad, shape, has_grad, other_grad
        else:
            return grad, shape, has_grad

class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)


if __name__ == '__main__':

    # fully shared network test
    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = TestNet()
    y_pred = net(x)
    pc_adam = GMS(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

    print('-' * 80)
    # seperated shared network test

    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = MultiHeadTestNet()
    y_pred_1, y_pred_2 = net(x)
    pc_adam = GMS(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.MSELoss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)