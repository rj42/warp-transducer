import numpy as np
import torch
from torch.autograd import Function, Variable
from torch.nn import Module

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "gradients only computed for acts - please " \
        "mark other tensors as not requiring gradients"


NEG_INF = -float("inf")


def forward_pass(log_probs, labels, blank, monotonic):
    T, U, _ = log_probs.shape
    alphas = np.zeros((T, U), dtype='f')
    m = 1 if monotonic else 0

    for t in range(1, T):
        alphas[t, 0] = alphas[t-1, 0] + log_probs[t-1, 0, blank]

    for u in range(1, U):
        if monotonic:
            alphas[0, u] = NEG_INF
        else:
            alphas[0, u] = alphas[0, u-1] + log_probs[0, u-1, labels[u-1]]

    for t in range(1, T):
        for u in range(1, U):
            no_emit = alphas[t-1, u] + log_probs[t-1, u, blank]
            emit = alphas[t-m, u-1] + log_probs[t-m, u-1, labels[u-1]]
            alphas[t, u] = np.logaddexp(emit, no_emit)

    loglike = alphas[T-1, U-1] + log_probs[T-1, U-1, blank]
    return alphas, loglike

def backward_pass(log_probs, labels, blank, monotonic):
    T, U, _ = log_probs.shape
    betas = np.zeros((T, U), dtype='f')
    betas[T-1, U-1] = log_probs[T-1, U-1, blank]
    m = 1 if monotonic else 0

    for t in reversed(range(T-1)):
        betas[t, U-1] = betas[t+1, U-1] + log_probs[t, U-1, blank]

    for u in reversed(range(U-1)):
        if monotonic:
            betas[T-1, u] = NEG_INF
        else:
            betas[T-1, u] = betas[T-1, u+1] + log_probs[T-1, u, labels[u]]

    for t in reversed(range(T-1)):
        for u in reversed(range(U-1)):
            no_emit = betas[t+1, u] + log_probs[t, u, blank]
            emit = betas[t+m, u+1] + log_probs[t, u, labels[u]]
            betas[t, u] = np.logaddexp(emit, no_emit)

    return betas, betas[0, 0]

def compute_gradient(log_probs, alphas, betas, labels, blank, fastemit_lambda, monotonic):
    T, U, _ = log_probs.shape
    grads = np.full(log_probs.shape, -float("inf"))
    m = 1 if monotonic else 0
    log_like = betas[0, 0]

    grads[T-1, U-1, blank] = alphas[T-1, U-1]
    grads[:T-1, :, blank] = alphas[:T-1, :] + betas[1:, :]

    for u, l in enumerate(labels):
        for t in range(T):
            if t + m == T:
                continue
            grads[t, u, l] = alphas[t, u] + betas[t+m, u+1]

    grads = -np.exp(grads + log_probs - log_like)

    # FastEmit regularization (https://arxiv.org/abs/2010.11148).
    if fastemit_lambda != 0:
        for u, l in enumerate(labels):
            grads[:, u, l] = (1.0 + fastemit_lambda) * grads[:, u, l]

    return grads

def transduce(log_probs, labels, blank=0, fastemit_lambda=0, monotonic=False):
    """
    Args:
        log_probs: 3D array with shape
              [input len, output len + 1, vocab size]
        labels: 1D array with shape [output time steps]
    Returns:
        float: The negative log-likelihood
        3D array: Gradients with respect to the
                    unnormalized input actications
    """
    alphas, ll_forward = forward_pass(log_probs, labels, blank, monotonic)
    betas, ll_backward = backward_pass(log_probs, labels, blank, monotonic)
    grads = compute_gradient(log_probs, alphas, betas, labels, blank, fastemit_lambda, monotonic)
    return -ll_forward, grads

def transduce_batch(log_probs, labels, flen, glen, blank=0, fastemit_lambda=0, monotonic=False):
    grads = np.zeros_like(log_probs)
    costs = []

    # TODO parallel loop
    for b in range(log_probs.shape[0]):
        t = int(flen[b])
        u = int(glen[b]) + 1
        ll, g = transduce(log_probs[b, :t, :u, :], labels[b, :u-1], blank,
            fastemit_lambda=fastemit_lambda, monotonic=monotonic
        )
        grads[b, :t, :u, :] = g
        costs.append(ll)
    return costs, grads

class _RNNT(Function):
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens):
        is_cuda = True if acts.is_cuda else False
        costs, grads = transduce_batch(
            acts.detach().cpu().numpy(), labels.cpu().numpy(),
            act_lens.cpu().numpy(), label_lens.cpu().numpy(),
            fastemit_lambda=0,
            monotonic=False,
        )

        costs = torch.FloatTensor([sum(costs)])
        grads = torch.Tensor(grads).to(acts)
        ctx.grads = Variable(grads)
        return costs

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grads, None, None, None

class RNNTLoss(Module):
    """
    Parameters:
        `size_average` (bool): normalize the loss by the batch size
                (default `False`)
        `blank_label` (int): default 0
    """
    def __init__(self):
        super(RNNTLoss, self).__init__()
        self.rnnt = _RNNT.apply

    def forward(self, acts, labels, act_lens, label_lens):
        assert len(labels.size()) == 2
        _assert_no_grad(labels)
        _assert_no_grad(act_lens)
        _assert_no_grad(label_lens)

        acts = torch.nn.functional.log_softmax(acts, -1)
        return self.rnnt(acts, labels, act_lens, label_lens)
