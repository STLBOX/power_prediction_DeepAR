def SMAPE(y_pred, target):
    """
    Symmetric mean absolute percentage. Assumes ``y >= 0``.
    Defined as ``2*(y - y_pred).abs() / (y.abs() + y_pred.abs())``
    """
    loss = (2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)).mean(0)*100
    return loss


def MAPE(y_pred, target):
    """
    Mean absolute percentage. Assumes ``y >= 0``.
    Defined as ``(y - y_pred).abs() / y.abs()``
    """
    loss = ((y_pred - target).abs() / (target.abs() + 1e-8)).mean(0)*100
    return loss


def MAE(y_pred, target):
    """
    Mean average absolute error.
    Defined as ``(y_pred - target).abs()``
    """
    loss = (target - y_pred).abs().mean(0)
    return loss


def MSE(y_pred, target):
    """
    Mean average absolute error.
    Defined as ``(y_pred - target).abs()``
    """
    loss = ((target - y_pred)**2).mean(0)
    return loss


def RMSE(y_pred, target):
    """
    Mean average absolute error.
    Defined as ``(y_pred - target).abs()``
    """
    loss = torch.sqrt(((target - y_pred)**2).mean(0))
    return loss