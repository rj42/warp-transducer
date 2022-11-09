#pragma once

#include "rnnt_helper.h"

template<typename T>
inline __device__ T logp(const T* const denom, const T* const acts, const int maxT, const int maxU, const int alphabet_size, int mb, int t, int u, int v) {
    const int col = (mb * maxT + t) * maxU + u;
    return denom[col] + acts[col * alphabet_size + v];
}

template<typename Tp>
__global__ void compute_alphas_kernel(const Tp* const acts, const Tp* const denom, Tp* alphas, Tp* llForward, const int* const xlen, const int* const ylen,
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    // launch B blocks, each block has U threads
    int b = blockIdx.x; // batch
    int u = threadIdx.x; // label id, u
    const int T = xlen[b];
    const int U = ylen[b] + 1;
    const int* labels = mlabels + b * (maxU - 1); // mb label start point
    const int offset = b * maxT * maxU;
    alphas += offset;
    if (u == 0) alphas[0] = 0;

    __syncthreads();
    for (int n = 1; n < T+U-1; ++n) {
        int t = n - u;
        if (u == 0) {
            if (t > 0 && t < T) {
                alphas[t * maxU + u] = alphas[(t-1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, b, t-1, 0, blank_);
            }
        } else if (u < U) {
            if (t == 0)
                alphas[u] = alphas[u-1] + logp(denom, acts, maxT, maxU, alphabet_size, b, 0, u-1, labels[u-1]);
            else if (t > 0 && t < T) {
                Tp no_emit = alphas[(t-1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, b, t-1, u, blank_);
                Tp emit = alphas[t * maxU + u-1] + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u-1, labels[u-1]);
                alphas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
        __syncthreads();
    }

    if (u == 0) {
        Tp loglike = alphas[(T-1) * maxU + U-1] + logp(denom, acts, maxT, maxU, alphabet_size, b, T-1, U-1, blank_);
        llForward[b] = loglike;
    }
}

template<typename Tp>
__global__ void compute_alphas_kernel_naive(const Tp* const acts, const Tp* const denom, Tp* alphas, Tp* llForward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    int tid = threadIdx.x; // mb
    const int T = xlen[tid];
    const int U = ylen[tid] + 1;
    const int* labels = mlabels + tid * (maxU - 1); // mb label start point
    const int offset = tid * maxT * maxU;
    alphas += offset;
    alphas[0] = 0;

    for (int t = 0; t < T; ++t) {
        for (int u = 0; u < U; ++u) {
            if (u == 0 && t > 0)
                alphas[t * maxU + u] = alphas[(t-1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t-1, 0, blank_);
            if (t == 0 && u > 0)
                alphas[u] = alphas[u-1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, 0, u-1, labels[u-1]);
            if (t > 0 && u > 0) {
                Tp no_emit = alphas[(t-1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t-1, u, blank_);
                Tp emit = alphas[t * maxU + u-1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t, u-1, labels[u-1]);
                alphas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
    }

    Tp loglike = alphas[(T-1) * maxU + U-1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, T-1, U-1, blank_);
    llForward[tid] = loglike;
}


template<typename Tp>
__global__ void compute_betas_kernel(const Tp* const acts, const Tp* const denom, Tp* betas, Tp* llBackward, const int* const xlen, const int* const ylen,
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    int b = blockIdx.x; // batch
    int u = threadIdx.x; // label id, u
    const int T = xlen[b];
    const int U = ylen[b] + 1;
    const int* labels = mlabels + b * (maxU - 1);
    const int offset = b * maxT * maxU;
    betas += offset;
    if (u == 0)
        betas[(T-1) * maxU + U-1] = logp(denom, acts, maxT, maxU, alphabet_size, b, T-1, U-1, blank_);

    __syncthreads();
    for (int n = T+U-2; n >= 0; --n) {
        int t = n - u;
        if (u == U-1) {
            if (t >= 0 && t < T-1) {
                betas[t * maxU + U-1] = betas[(t+1) * maxU + U-1] + logp(denom, acts, maxT, maxU, alphabet_size, b, t, U-1, blank_);
            }
        } else if (u < U-1) {
            if (t == T-1) {
                betas[(T-1) * maxU + u] = betas[(T-1) * maxU + u+1] + logp(denom, acts, maxT, maxU, alphabet_size, b, T-1, u, labels[u]);
            } else if (t >= 0 && t < T-1) {
                Tp no_emit = betas[(t+1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, blank_);
                Tp emit = betas[t * maxU + u+1] + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, labels[u]);
                betas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
        __syncthreads();
    }

    if (u == 0) {
        llBackward[b] = betas[0];
    }
}

template<typename Tp>
__global__ void compute_betas_kernel_naive(const Tp* const acts, const Tp* const denom, Tp* betas, Tp* llBackward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    int tid = threadIdx.x; // mb
    const int T = xlen[tid];
    const int U = ylen[tid] + 1;
    const int* labels = mlabels + tid * (maxU - 1);
    const int offset = tid * maxT * maxU;
    betas += offset;
    betas[(T-1) * maxU + U-1] = logp(denom, acts, maxT, maxU, alphabet_size, tid, T-1, U-1, blank_);

    for (int t = T-1; t >=0; --t) {
        for (int u = U-1; u >= 0; --u) {
            if (u == U-1 && t < T-1)
                betas[t * maxU + U-1] = betas[(t+1) * maxU + U-1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t, U-1, blank_);
            if (t == T-1 && u < U-1)
                betas[(T-1) * maxU + u] = betas[(T-1) * maxU + u+1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, T-1, u, labels[u]);
            if (t < T-1 && u < U-1) {
                Tp no_emit = betas[(t+1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t, u, blank_);
                Tp emit = betas[t * maxU + u+1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t, u, labels[u]);
                betas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
    }

    llBackward[tid] = betas[0];
}

template<int NT, typename Tp>
__global__ void compute_grad_kernel(Tp* grads, const Tp* const acts, const Tp* const denom, const Tp* alphas, const Tp* betas, const Tp* const logll, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_, const float fastemit_lambda_) {
    int tid = threadIdx.x; // alphabet dim
    int idx = tid;
    int col = blockIdx.x; // mb, t, u

    int u = col % maxU;
    int bt = (col - u) / maxU;
    int t = bt % maxT;
    int mb = (bt - t) / maxT;

    const int T = xlen[mb];
    const int U = ylen[mb] + 1;
    const int* labels = mlabels + mb * (maxU - 1);

    // See: https://github.com/titu1994/warprnnt_numba/blob/b1bc81e02dfb05143c3d55ac7b50c8131e85b194/warprnnt_numba/rnnt_loss/utils/cuda_utils/gpu_rnnt_kernel.py#L264
    //
    // b(t + ?, u) = b(t + 1, u) if monotonic_loss else b(t, u)
    //
    // d(logsoftmax(x_i), x_k) = δik - p_k, where p_k = softmax(x_k)
    //
    // d(L(logsoftmax(x)), x_k) = Σdi(L) * d(logsoftmax(x_i), x_k) = Σdi(L) * (δik - p_k) = Σg_i * (δik - p_k)
    //
    // d(L, log(p(i|t, u))) = -a(t, u) * p(i|t, u) / p(y*|x) * (I(i == y_u+1) * (1 + λ) * b(t + ?, u + 1)) + I(i == ∅) * b(t + 1, u)) = g_i
    //
    // y(t, u) = p(y_u+1 | t, u)
    // ∅(t, u) = p(∅ | t, u)
    //
    // d(L, x_k) =
    //      -a(t, u) * p(k|t, u) / p(y*|x) * (
    //          - (1 + λ) * b(t + ?, u + 1) * y_u + b(t + 1, u) * p_∅)                 // expanded sum of I(k == y_u+1) and I(k == ∅) with p_k
    //          + (1 + λ) * I(k == y_u+1) * b(t + ?, u + 1)) + I(k == ∅) * b(t + 1, u) // expanded sum with δik
    //     ) =
    //      -a(t, u) * p(k|t, u) / p(y*|x) * (
    //          - b(t, u)                                                 // by definiton of b(t, u)
    //          - λ * b(t +?, u + 1) * y_u
    //          + (1 + λ) * I(k == y_u+1) * b(t + ?, u + 1)) + I(k == ∅) * b(t + 1, u)
    //        )
    //
    // coeff               = a(t, u) * p(k|t, u) / p(y*|x)
    // base                = coeff * (b(t, u) + λ * b(t + ?, u + 1) * y_u)
    // grad for non-labels = base - coeff * b(t + 1, u)
    // grad for labels     = base - coeff * b(t + ?, u + 1) * (1 + λ)
    //
    //
    if (t < T && u < U) {
        while (idx < alphabet_size) {
            Tp logpk = denom[col] + acts[col * alphabet_size + idx];
            // Tp logpk = logp(denom, acts, maxT, maxU, alphabet_size, mb, t, u, idx);
            // logpk = denom[b, t, u] + acts[b, t, u, v]
            Tp coeff = alphas[col] + logpk - logll[mb];
            Tp grad = exp(coeff + betas[col]);

            Tp fastemit_grad = 0.0;
            if (fastemit_lambda_ != 0.0 && u < U - 1) {
                // If FastEmit regularization is enabled, calculate the gradient of probability of predicting the next label
                // at the current timestep.
                // The formula for this is Equation 9 in https://arxiv.org/abs/2010.11148, multiplied by the log probability
                // of the current step (t, u), normalized by the total log likelihood.
                // Once the gradient has been calculated, scale it by `fastemit_lambda`, as in Equation 10.
                fastemit_grad = fastemit_lambda_ * exp(
                    coeff
                    + (denom[col] + acts[col * alphabet_size + labels[u]])  // y_hat(t, u) = P(labels[u]| t, u)
                    + betas[col + 1]  // betas(t, u+1)
                );

                // Update the gradient of act[b, t, u, v] with the gradient from FastEmit regularization
                grad += fastemit_grad;
            }

            // grad to last blank transition
            // grad[b, T-1, U-1, v=blank] -= exp(alphas[b, t, u) + logpk - logll[b])
            if (idx == blank_ && t == T-1 && u == U-1) {
                grad -= exp(coeff);
            }

            // grad[b, t<T-1, u, v=blank] -= exp(alphas[b, t, u] + logpk - logll[b] + betas[b, t + 1, u])
            if (idx == blank_ && t < T-1) {
                grad -= exp(coeff + betas[col + maxU]);
            }

            // grad of correct token across u < U;
            // grad[b, t, u<U-1, v=label[u]] -= exp(alphas[b, t, u] + logpk - logll[b] + betas[b, t, u+1])
            // Scale the gradient by (1.0 + FastEmit_lambda) in log space, then exponentiate
            if (u < U-1 && idx == labels[u]) {
                // exp(log(1 + fastemit_lambda) + ...) is numerically more stable than
                // multiplying (1.0 + fastemit_lambda) with result.
                grad -= exp(log1p(fastemit_lambda_) + coeff + betas[col+1]);
            }
            grads[col * alphabet_size + idx] = grad;

            idx += NT;
        }
    }
}
