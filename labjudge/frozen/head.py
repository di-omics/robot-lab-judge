"""The small head: standardize, then a class-balanced logistic regression with an L2 penalty. Numpy only."""

import numpy as np

L2_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
N_FOLDS = 5


class Scaler:
    """Mean and standard deviation from the rows it is fitted on, and nothing else."""

    def __init__(self, x: np.ndarray):
        x = np.asarray(x, np.float64)
        self.mean = x.mean(axis=0)
        std = x.std(axis=0)
        self.std = np.where(std > 1e-8, std, 1.0)
        self.n_fit = len(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, np.float64) - self.mean) / self.std


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """Weight n / (2 n_class) per row, so each class carries half of the loss."""
    y = np.asarray(y, int)
    n, n1 = len(y), int(y.sum())
    n0 = n - n1
    if n0 == 0 or n1 == 0:
        return np.ones(n)
    return np.where(y == 1, n / (2 * n1), n / (2 * n0))


def _sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))


def _objective(c, b, z_x, y, s, l2):
    z = z_x @ c + b
    loss = np.logaddexp(0, z) - y * z
    return float(s @ loss + 0.5 * l2 * c @ c)


def _newton(x, y, s, l2, max_iter=200, tol=1e-8):
    """Minimize sum_i s_i * logloss_i + l2/2 * |c|^2 over (c, b); the bias b is not penalized."""
    n, d = x.shape
    c = np.zeros(d)
    b = 0.0
    obj = _objective(c, b, x, y, s, l2)
    xb = np.hstack([x, np.ones((n, 1))])
    pen = np.full(d + 1, l2)
    pen[-1] = 0.0
    for _ in range(max_iter):
        p = _sigmoid(x @ c + b)
        g = xb.T @ (s * (p - y)) + pen * np.append(c, 0.0)
        if np.max(np.abs(g)) < tol * max(1.0, s.sum()):
            break
        h = (xb * (s * p * (1 - p))[:, None]).T @ xb + np.diag(pen) + 1e-10 * np.eye(d + 1)
        step = np.linalg.solve(h, g)
        t = 1.0
        while True:
            c_new, b_new = c - t * step[:-1], b - t * step[-1]
            obj_new = _objective(c_new, b_new, x, y, s, l2)
            if obj_new <= obj - 1e-4 * t * (g @ step) or t < 1e-10:
                break
            t *= 0.5
        if obj - obj_new < 1e-12 * max(1.0, abs(obj)):
            c, b, obj = c_new, b_new, obj_new
            break
        c, b, obj = c_new, b_new, obj_new
    return c, b


def fit_logreg(x: np.ndarray, y: np.ndarray, l2: float, sample_weight=None) -> tuple[np.ndarray, float]:
    """Weights w and bias b of an L2 logistic regression on already standardized x.

    The best w lies in the span of the rows of x (anything outside it adds penalty and no fit),
    so with fewer rows than columns the problem is solved exactly in that span, found by an SVD.
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    s = balanced_weights(y) if sample_weight is None else np.asarray(sample_weight, np.float64)
    n, d = x.shape
    if n < d:
        _, sv, vt = np.linalg.svd(x, full_matrices=False)
        keep = sv > sv.max() * 1e-10
        v = vt[keep].T  # d x r, orthonormal columns
        c, b = _newton(x @ v, y, s, l2)
        return v @ c, b
    return _newton(x, y, s, l2)


def predict_proba(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return _sigmoid(np.asarray(x, np.float64) @ w + b)


def grouped_folds(groups, k: int = N_FOLDS, seed: int = 0) -> np.ndarray:
    """Fold number per row. Every row of a group (a layout seed) lands in the same fold.

    The sorted unique groups are shuffled with a fixed seed and dealt out in turn.
    """
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {g: i % k for i, g in zip(range(len(uniq)), uniq[order])}
    return np.array([fold_of[g] for g in groups.tolist()])


def balanced_accuracy(y, pred) -> float:
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    rec = [np.mean(pred[y == c] == c) for c in (0, 1) if np.any(y == c)]
    return float(np.mean(rec)) if rec else float("nan")


def auroc(y, p) -> float:
    """Area under the ROC curve from ranks, ties counted as half."""
    y = np.asarray(y, int)
    p = np.asarray(p, np.float64)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i: j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cv_oof(x: np.ndarray, y: np.ndarray, folds: np.ndarray, l2: float) -> np.ndarray:
    """Out-of-fold probabilities. The scaler is fitted on the training folds only, inside each fold."""
    p = np.full(len(y), np.nan)
    for f in sorted(set(folds.tolist())):
        tr, va = folds != f, folds == f
        sc = Scaler(x[tr])
        w, b = fit_logreg(sc(x[tr]), y[tr], l2)
        p[va] = predict_proba(sc(x[va]), w, b)
    return p
