import numpy as np
import time


def boundaryCheck(Positions, lb, ub):
    for i in range(Positions.shape[0]):
        Flag4ub = Positions[i, :] > ub
        Flag4lb = Positions[i, :] < lb
        Positions[i, :] = (Positions[i, :] * (~(Flag4ub[i, :] + Flag4lb[i, :]))) + ub[i, :] * Flag4ub[i, :] + lb[i,
                                                                                                              :] * Flag4lb[
                                                                                                                   i, :]
    return Positions


# Gorilla Troops Optimizer (GTO)
def GTO(X, fobj, lower_bound, upper_bound, max_iter):
    pop_size, variables_no = X.shape
    ub = upper_bound[0, :]
    lb = lower_bound[0, :]
    # initialize Silverback
    Silverback = []
    Silverback_Score = float('inf')
    convergence_curve = np.zeros(max_iter)
    Pop_Fit = np.zeros((pop_size))
    for i in np.arange(pop_size):
        Pop_Fit[i] = fobj(X[i, :])
        if Pop_Fit[i] < Silverback_Score:
            Silverback_Score = Pop_Fit[i]
            Silverback = X[i, :]

    GX = X
    ##  Controlling parameter
    p = 0.03
    Beta = 3
    w = 0.8
    t = 0

    ct = time.time()
    # Main loop
    for It in np.arange(max_iter):
        a = (np.cos(2 * np.random.rand()) + 1) * (1 - It / max_iter)
        C = a * (2 * np.random.rand() - 1)
        # Exploration:
        for i in np.arange(pop_size):
            if np.random.rand() < p:
                GX[i, :] = (ub - lb) * np.random.rand() + lb
            else:
                if np.random.rand() >= 0.5:
                    Z = np.random.uniform(- a, a, variables_no)
                    H = np.multiply(Z, X[i, :])
                    GX[i, :] = (np.random.rand() - a) * X[np.random.randint(np.array([pop_size])), :] + np.multiply(
                        C, H)
                else:
                    GX[i, :] = X[i, :] - np.multiply(C, (
                            C * (X[i, :] - GX[np.random.randint(np.array([pop_size])), :]) + np.random.rand() * (
                            X[i, :] - GX[
                                      np.random.randint(np.array([pop_size])), :])))
        GX = boundaryCheck(GX, lower_bound, upper_bound)
        # Group formation operation
        for i in np.arange(pop_size):
            New_Fit = fobj(GX[i, :])
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :]
            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i, :]
        # Exploitation:
        for i in np.arange(pop_size):
            if a >= w:
                g = 2 ** C
                delta = (np.abs(np.mean(GX)) ** g) ** (1 / g)
                GX[i, :] = np.multiply(C * delta, (X[i, :] - Silverback)) + X[i, :]
            else:
                if np.random.rand() >= 0.5:
                    h = np.random.randn(1, variables_no)
                else:
                    h = np.random.randn(1, 1)
                r1 = np.random.rand()
                GX[i, :] = Silverback - np.multiply(
                    (Silverback * (2 * r1 - 1) - X[i, :] * (2 * r1 - 1)), (Beta * h))
        GX = boundaryCheck(GX, lower_bound, upper_bound)
        # Group formation operation
        for i in np.arange(pop_size):
            New_Fit = fobj(GX[i, :])
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :]
            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i, :]
        convergence_curve[It] = Silverback_Score
    ct = time.time() - ct
    return Silverback_Score, convergence_curve, Silverback, ct
