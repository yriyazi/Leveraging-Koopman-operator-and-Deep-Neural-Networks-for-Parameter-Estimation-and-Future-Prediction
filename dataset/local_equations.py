def lorenz(t, state, sigma = 10, rho = 28, beta = 8/3):
    # Parameters
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]