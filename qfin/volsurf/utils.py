import numpy as np
from scipy.optimize import newton


def raw_to_svi_jw(ttm, a, b, rho, m, sigma):
    v_t = (a + b * (-rho * m + np.sqrt(m ** 2 + sigma ** 2))) / ttm
    w_t = v_t * ttm
    psi_t = b / 2 / np.sqrt(w_t) * (-m / np.sqrt(m ** 2 + sigma ** 2) + rho)
    p_t = b * (1 - rho) / np.sqrt(w_t)
    c_t = b * (1 + rho) / np.sqrt(w_t)
    v_tilde_t = (a + b * sigma * np.sqrt(1 - sigma ** 2)) / ttm
    return v_t, psi_t, p_t, c_t, v_tilde_t


def svi_jw_to_raw(ttm, v_t, psi_t, p_t, c_t, v_tilde_t):
    w_t = v_t * ttm
    b = np.sqrt(w_t) / 2 * (c_t + p_t)
    rho = 1 - p_t * np.sqrt(w_t) / b
    beta = rho - 2 * psi_t * np.sqrt(w_t) / b
    alpha = np.sign(beta) * np.sqrt(1 / beta ** 2 - 1)
    m = (v_t - v_tilde_t) * ttm / b / (-rho * np.sign(alpha) * np.sqrt(1 + alpha ** 2) - alpha * np.sqrt(1 - rho ** 2))
    sigma = alpha * m
    a = v_tilde_t * ttm - b * sigma * np.sqrt(1 - rho ** 2)
    return a, b, rho, m, sigma


def raw_total_variance(xs, a, b, rho, m, sigma):
    return a + b * (rho * (xs - m) + np.sqrt((xs - m) ** 2 + sigma ** 2))


def get_crossings(x0, x1, cutoff=2.):

    f = lambda x: raw_total_variance(x, *x0) - raw_total_variance(x, *x1)

    try:
        # there are at most 4 crossing possible for two convex curves
        crossings = newton(f, np.linspace(-cutoff, cutoff, 5), disp=False)
    except RuntimeError:
        return np.array([])

    # filter out those which are indeed roots, sometimes solver terminates in a non-zero
    crossings = np.array([x for x in crossings if np.isclose(f(x), 0, atol=1e-3)])

    # filter out those which are in the range
    crossings = np.array([x for x in crossings if -cutoff < x < cutoff])

    # filter out duplicates
    crossings = np.array(list(set(np.round([x for x in crossings], 4))))

    return crossings
