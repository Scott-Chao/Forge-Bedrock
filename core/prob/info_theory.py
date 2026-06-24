"""Information theory primitives: Entropy, KL Divergence, Cross-Entropy.

These three quantities form the theoretical bridge between probability
distributions and loss functions — the core insight of Phase 3.

    Entropy:       H(p)       = -∑ p(x) log p(x)
    KL Divergence: D_KL(p||q) =  ∑ p(x) log(p(x) / q(x))
    Cross-Entropy: H(p, q)    = -∑ p(x) log q(x)   = H(p) + D_KL(p||q)

All functions accept empirical distributions (raw data) or theoretical
distributions (from core.prob.distributions).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def entropy(p: np.ndarray, base: float = np.e) -> float:
    """Compute the Shannon entropy H(p) = -∑ p_i * log_base(p_i).

    Parameters
    ----------
    p : np.ndarray
        Probability vector(s). Must be non-negative and sum to 1.
        If 2-D, each row is treated as a separate distribution.
    base : float
        Logarithm base. Use np.e for nats, 2 for bits, 10 for dits.

    Returns
    -------
    float or np.ndarray
        Entropy value(s). Returns 0-d array for 1-D input, 1-D for 2-D.
    """
    p = np.asarray(p)
    log_p = np.log(p, where=p > 0, out=np.zeros_like(p))
    H = -np.sum(p * log_p, axis=-1) / np.log(base)
    return H


def entropy_from_counts(counts: np.ndarray, base: float = np.e) -> float:
    """Compute entropy from raw counts instead of probabilities."""
    counts = np.asarray(counts)
    total = np.sum(counts)
    if total == 0:
        return 0.0
    p = counts / total
    return entropy(p, base=base)


# ---------------------------------------------------------------------------
# KL Divergence
# ---------------------------------------------------------------------------


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    base: float = np.e,
) -> float | np.ndarray:
    """Kullback-Leibler divergence D_KL(p || q) = ∑ p_i * log(p_i / q_i).

    Parameters
    ----------
    p, q : np.ndarray
        Probability vectors. Same shape. Non-negative, sum to 1.
        If 2-D, each row is compared row-wise.
    base : float
        Logarithm base.

    Returns
    -------
    float or np.ndarray
        KL divergence(s). Always >= 0, 0 iff p == q element-wise.
    """
    p, q = np.asarray(p), np.asarray(q)
    ratio = np.divide(p, q, where=p > 0, out=np.zeros_like(p))
    log_ratio = np.log(ratio, where=p > 0, out=np.zeros_like(ratio))
    D_KL = np.sum(p * log_ratio, axis=-1) / np.log(base)
    return D_KL


def js_divergence(
    p: np.ndarray,
    q: np.ndarray,
    base: float = np.e,
) -> float | np.ndarray:
    """Jensen-Shannon divergence: symmetric, bounded version of KL.

    JS(p||q) = 0.5 * D_KL(p || m) + 0.5 * D_KL(q || m)
    where m = 0.5 * (p + q)
    """
    p, q = np.asarray(p), np.asarray(q)
    m = 0.5 * (p + q)
    D_JS = 0.5 * kl_divergence(p, m, base=base) + 0.5 * kl_divergence(q, m, base=base)
    return D_JS


# ---------------------------------------------------------------------------
# Cross-Entropy
# ---------------------------------------------------------------------------


def cross_entropy(
    p: np.ndarray,
    q: np.ndarray,
    base: float = np.e,
) -> float | np.ndarray:
    """Cross-entropy H(p, q) = -∑ p_i * log(q_i).

    Parameters
    ----------
    p, q : np.ndarray
        Probability vectors. Same shape. 2-D inputs are compared row-wise.
    base : float
        Logarithm base.

    Returns
    -------
    float or np.ndarray
        Cross-entropy value(s). Always >= H(p), equal iff p == q.
    """
    p, q = np.asarray(p), np.asarray(q)
    log_q = np.log(q, where=p > 0, out=np.zeros_like(q))
    H = -np.sum(p * log_q, axis=-1) / np.log(base)
    return H


# ---------------------------------------------------------------------------
# Mutual Information (bonus — useful in Phase 4 feature selection)
# ---------------------------------------------------------------------------


def mutual_information(
    joint: np.ndarray,
    base: float = np.e,
) -> float:
    """Mutual information I(X; Y) for a 2-D joint probability table.

    I(X; Y) = H(X) + H(Y) - H(X, Y) = D_KL(p(x,y) || p(x)p(y))

    Parameters
    ----------
    joint : np.ndarray
        2-D joint probability table p(x, y), shape (K, L).
        Must be non-negative and sum to 1.
    """
    p_x, p_y = joint.sum(axis=1), joint.sum(axis=0)
    # Flatten joint for scalar joint-entropy (entropy on 2-D is row-wise)
    info = (
        entropy(p_x, base=base)
        + entropy(p_y, base=base)
        - entropy(joint.ravel(), base=base)
    )
    return float(info)
