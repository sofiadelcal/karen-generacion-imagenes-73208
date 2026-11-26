"""Red adversarial mínima en NumPy para los ejercicios reproducibles.

Las capas y la actualización se dejan explícitas para poder seguir el cálculo.
No se usa ninguna etiqueta durante el entrenamiento adversarial.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


class DenseNet:
    def __init__(self, widths, output="linear", seed=0):
        rng = np.random.default_rng(seed)
        self.w = [rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float64)
                  for a, b in zip(widths[:-1], widths[1:])]
        self.b = [np.zeros((1, b)) for b in widths[1:]]
        self.output = output
        self.m = [np.zeros_like(v) for v in self.w + self.b]
        self.v = [np.zeros_like(v) for v in self.w + self.b]
        self.t = 0

    def forward(self, x):
        values = [x]
        pre = []
        for i, (w, b) in enumerate(zip(self.w, self.b)):
            z = values[-1] @ w + b
            pre.append(z)
            if i == len(self.w) - 1:
                y = np.tanh(z) if self.output == "tanh" else z
            else:
                y = np.maximum(z, 0.2 * z)
            values.append(y)
        return values[-1], (values, pre)

    def backward(self, grad, cache):
        values, pre = cache
        gw, gb = [None] * len(self.w), [None] * len(self.b)
        for i in reversed(range(len(self.w))):
            if i == len(self.w) - 1 and self.output == "tanh":
                grad = grad * (1 - values[i + 1] ** 2)
            elif i < len(self.w) - 1:
                grad = grad * np.where(pre[i] > 0, 1.0, 0.2)
            gw[i] = values[i].T @ grad
            gb[i] = grad.sum(axis=0, keepdims=True)
            grad = grad @ self.w[i].T
        return grad, gw + gb

    def step(self, grads, lr=0.0005, beta1=0.5, beta2=0.999):
        self.t += 1
        for p, g, m, v in zip(self.w + self.b, grads, self.m, self.v):
            m *= beta1
            m += (1 - beta1) * g
            v *= beta2
            v += (1 - beta2) * g * g
            p -= lr * (m / (1 - beta1 ** self.t)) / (
                np.sqrt(v / (1 - beta2 ** self.t)) + 1e-8)


def train(real_samples, noise_dim, g_widths, d_widths, steps, batch, seed, lr):
    rng = np.random.default_rng(seed)
    g = DenseNet([noise_dim, *g_widths, real_samples.shape[1]],
                 output="tanh" if real_samples.shape[1] == 64 else "linear",
                 seed=seed + 1)
    d = DenseNet([real_samples.shape[1], *d_widths, 1], seed=seed + 2)
    probe = rng.normal(size=(256, noise_dim))
    initial = g.forward(probe)[0].copy()
    records = []
    for step in range(1, steps + 1):
        real = real_samples[rng.integers(len(real_samples), size=batch)]
        noise = rng.normal(size=(batch, noise_dim))
        fake = g.forward(noise)[0]
        xr, cr = d.forward(real)
        xf, cf = d.forward(fake)
        pr, pf = sigmoid(xr), sigmoid(xf)
        # Suavizado de la etiqueta real; las falsas conservan la etiqueta cero.
        _, gr = d.backward((pr - 0.9) / batch, cr)
        _, gf = d.backward(pf / batch, cf)
        d.step([a + b for a, b in zip(gr, gf)], lr=lr)

        noise = rng.normal(size=(batch, noise_dim))
        fake, cg = g.forward(noise)
        logit, cd = d.forward(fake)
        p = sigmoid(logit)
        input_grad, _ = d.backward((p - 1) / batch, cd)
        _, gg = g.backward(input_grad, cg)
        g.step(gg, lr=lr)
        if step % 50 == 0 or step == 1:
            ld = -np.mean(0.9 * np.log(pr + 1e-9) + 0.1 * np.log(1 - pr + 1e-9)
                          + np.log(1 - pf + 1e-9))
            lg = -np.mean(np.log(p + 1e-9))
            records.append((step, float(ld), float(lg)))
    final = g.forward(probe)[0]
    return initial, final, np.asarray(records), g


def mmd_rbf(a, b, bandwidth):
    def kernel(x, y):
        d2 = np.maximum(((x[:, None, :] - y[None, :, :]) ** 2).sum(2), 0)
        return np.exp(-d2 / (2 * bandwidth ** 2))
    aa, bb, ab = kernel(a, a), kernel(b, b), kernel(a, b)
    return float(aa.mean() + bb.mean() - 2 * ab.mean())
