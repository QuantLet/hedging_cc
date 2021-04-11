import numpy as np
from scipy.optimize import minimize


def calibrate(model, market_vs, x0, reg, stream=None, **kwargs):

    if stream is not None:
        ps = ",".join(model.labels)
        stream.write(f"cost,rmse,penalty,{ps}\n")

    def distance(x):
        model_vs = model.vs(x, market_vs.points)
        rmse = market_vs - model_vs
        penalty = np.dot(np.dot(np.transpose(x), reg), x)
        return rmse, penalty

    def cost(x):

        rmse, penalty = distance(x)
        cost = rmse + penalty

        if stream is not None:
            ps = ",".join(["{:.6f}"] * len(x)).format(*x)
            stream.write(f"{cost:.6f},{rmse:.6f},{penalty:.6f},{ps}\n")

        return cost

    # calibrate
    result = minimize(cost, x0=x0, bounds=model.bounds, method="SLSQP", **kwargs)

    # add rmse and penalty
    x1 = result['x']
    rmse, penalty = distance(x1)

    result['rmse'] = rmse
    result['penalty'] = penalty

    return result
