def simulate_risk(pred, factor):
    """
    factor: simulate severity increase (0 to 1)
    """
    new_pred = pred + (1 - pred) * factor
    change = ((new_pred - pred) / pred) * 100 if pred != 0 else 0

    return new_pred, change