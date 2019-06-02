"""

@author:    Patrik Purgai
@copyright: Copyright 2019, swats
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.05.30.
"""

# pylint: disable=no-member

import torch


class SWATS(torch.optim.Optimizer):
    r"""Implements Switching from Adam to SGD technique. Proposed in
    `Improving Generalization Performance by Switching from Adam to SGD`
    by Nitish Shirish Keskar, Richard Socher (2017).

    The method applies Adam in the first phase of the training, then
    switches to SGD when a criteria is met.

    Implementation of Adam and SGD update are from `torch.optim.Adam` and
    `torch.optim.SGD`.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, verbose=False, 
                 nesterov=False):
        if not 0.0 <= lr:
            raise ValueError(
                "Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError(
                "Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, phase='ADAM',
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        verbose=verbose, nesterov=nesterov)

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('nesterov', False)
            group.setdefault('verbose', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): 
                A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for w in group['params']:
                if w.grad is None:
                    continue
                grad = w.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, '
                        'please consider SparseAdam instead')

                amsgrad = group['amsgrad']

                state = self.state[w]

                # state initialization
                if len(state) == 0:
                    state['step'] = 0
                    # exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(w.data)
                    # exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(w.data)
                    # moving average for the non-orthogonal projection scaling
                    state['exp_avg2'] = w.new(1).fill_(0)
                    if amsgrad:
                        # maintains max of all exp. moving avg.
                        # of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(w.data)

                exp_avg, exp_avg2, exp_avg_sq = \
                    state['exp_avg'], state['exp_avg2'], state['exp_avg_sq'],

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], w.data)

                # if its SGD phase, take an SGD update and continue
                if group['phase'] == 'SGD':
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(
                            grad).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(beta1).add_(grad)
                        grad = buf

                    grad.mul_(1 - beta1)
                    if group['nesterov']:
                        grad.add_(beta1, buf)

                    w.data.add_(-group['lr'], grad)
                    continue

                # decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # maintains the maximum of all 2nd
                    # moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * \
                    (bias_correction2 ** 0.5) / bias_correction1

                p = -step_size * (exp_avg / denom)
                w.data.add_(p)

                p_view = p.view(-1)
                pg = p_view.dot(grad.view(-1))

                if pg != 0:
                    # the non-orthognal scaling estimate
                    scaling = p_view.dot(p_view) / -pg
                    exp_avg2.mul_(beta2).add_(1 - beta2, scaling)

                    # bias corrected exponential average
                    corrected_exp_avg = exp_avg2 / bias_correction2

                    # checking criteria of switching to SGD training
                    if state['step'] > 1 and \
                            corrected_exp_avg.allclose(scaling, rtol=1e-6) and \
                            corrected_exp_avg > 0:
                        group['phase'] = 'SGD'
                        group['lr'] = corrected_exp_avg.item()
                        if group['verbose']:
                            print('Switching to SGD after '
                                  '{} steps with lr {:.5f} '
                                  'and momentum {:.5f}.'.format(
                                      state['step'], group['lr'], beta1))

        return loss
