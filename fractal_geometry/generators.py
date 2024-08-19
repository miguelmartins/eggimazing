import numpy as np
from tqdm import tqdm


def get_gaussian_white_noise_1d(length, mu=0., sigma=1.):
    # A process of the type X(t) = X(t-1) + Gaussian(mu, sigma^2)
    x = np.zeros(length, dtype=np.float32)
    white_noise = [x[t - 1] + np.random.normal(loc=mu, scale=sigma, ) / (length - 1)
                   for t in range(1, length)]
    return white_noise


def midpoint_brownian_noise_1d(*, max_level, sigma):
    # From "The Science of Fractal Images"
    # Chapter 2 - Section 2.2.4 - Random Displacement Method

    # Here we let the variance of each displacement be (1 / (2^(n+1))) *  sigma^2
    # We store it in array delta for convenience, but it could be calculated in the recurrence
    delta = np.array([sigma * (0.5 ** ((i + 1) / 2))
                      for i in range(1, max_level + 1)])
    n = 2 ** max_level  # Each recursion has 2 branches. Hence, we will have n = 2^max_level displacements
    x = np.zeros(n + 1,
                 dtype=np.float32)  # If we let the length be 2 ** (n+1) we will have even indices! This is for
    # convenience
    x[n] = np.random.normal(loc=0,
                            scale=sigma)  # X(0) := 0; X(1) = Normal(mu, sigma^2). Here we map this interval to

    # integers in [0, 2 ** max_level]

    def midpoint_recursion(x, lower, upper, level):
        midpoint = int((lower + upper) / 2)
        # See the recursion on section 2.2.4
        # I think we draw from a gaussian with sigma=1 since delta already has sigma in it.
        x[midpoint] = (0.5 * (x[lower] + x[upper])) + delta[level] * np.random.normal(loc=0, scale=1.)
        if level < max_level - 1:  # we are using indices at 0, not at 1, so we go to max_level-1
            midpoint_recursion(x, lower, midpoint, level + 1)
            midpoint_recursion(x, midpoint, upper, level + 1)

    midpoint_recursion(x, 0, n, 0)
    return x


def midpoint_fractional_brownian_noise_1d(*, max_level, sigma, hurst):
    delta = np.array([sigma * (0.5 ** (i * hurst)) * np.sqrt(0.5) * np.sqrt(1 - (2 ** ((2 * hurst) - 2)))
                      for i in range(1, max_level + 1)])
    n = 2 ** max_level  # Each recursion has 2 branches. Hence, we will have n = 2^max_level displacements
    x = np.zeros(n + 1,
                 dtype=np.float32)  # If we let the length be 2 ** (n+1) we will have even indices! This is for
    # convenience
    x[n] = np.random.normal(loc=0,
                            scale=sigma)  # X(0) := 0; X(1) = Normal(mu, sigma^2). Here we map this interval to

    # integers in [0, 2 ** max_level]

    def midpoint_recursion(x, lower, upper, level):
        midpoint = int((lower + upper) / 2)
        # See the recursion on section 2.2.4
        x[midpoint] = (0.5 * (x[lower] + x[upper])) + delta[level] * np.random.normal(loc=0, scale=1.)
        if level < max_level - 1:  # we are using indices at 0, not at 1, so we go to max_level-1
            midpoint_recursion(x, lower, midpoint, level + 1)
            midpoint_recursion(x, midpoint, upper, level + 1)

    midpoint_recursion(x, 0, n, 0)
    return x


def additions_midpoint_fractional_brownian_noise_1d(x, max_level, sigma, hurst):
    #
    # see sections 2.3.2 and 2.3.3
    # max_level + 2 because I suspect that the TO operator in the book is <=
    delta = np.array([sigma * (0.5 ** (i * hurst)) * np.sqrt(0.5) * np.sqrt(1 - (2 ** ((2 * hurst) - 2)))
                      for i in range(1, max_level + 2)])  # formula is special case at the end of section 2.3.3

    n = 2 ** (max_level)
    x[0] = 0.
    x[n] = np.random.normal(loc=0, scale=sigma)
    D = n
    d = int(D / 2)
    level = 1
    while level <= max_level:
        print(f'level {level}. D={D} d={d}')
        for i in range(d, n - d + 1, D):
            x[i] = 0.5 * (x[i - d] + x[i + d])
        for i in range(0, n + 1, d):
            x[i] = x[i] + delta[level] * np.random.normal(loc=0, scale=1)
        D = int(D / 2)
        d = int(d / 2)
        level = level + 1


def midpoint_fractional_colours(x, max_level, sigma, hurst, use_addition=False):
    # Direct implementation of Fractal Dimension of Colour Images by Ivanovici and Richard
    # over the code proposed in the book
    def f(delta_, x_):
        return np.mean(x_, axis=0) + delta_ * np.random.normal(loc=0, scale=1, size=(1, 3))

    n = 2 ** (max_level)
    delta = sigma
    # V(x1, x2) = [r, g, b]  # H = 0.1
    # v(x1, x2, r)  # H = 0.1
    # v(x1, x2, g)  # H = 0.1
    x[0, 0] = delta * np.random.normal(loc=0, scale=1, size=(1, 3))
    x[0, n] = delta * np.random.normal(loc=0, scale=1, size=(1, 3))
    x[n, 0] = delta * np.random.normal(loc=0, scale=1, size=(1, 3))
    x[n, n] = delta * np.random.normal(loc=0, scale=1, size=(1, 3))
    D = n
    d = int(n / 2)

    for stage in range(1, max_level + 1):
        # grid of type 1 to type 2
        delta = delta * (0.5 ** (0.5 * hurst))
        # interpolate and offset points
        for i in range(d, n - d + 1, D):
            for j in range(d, n - d + 1, D):
                x[i][j] = f(delta, [x[i + d][j + d],
                                    x[i + d][j - d],
                                    x[i - d][j + d],
                                    x[i - d][j - d]])

        if use_addition:
            for i in range(0, n + 1, D):
                for j in range(0, n + 1, D):
                    x[i][j] = x[i][j] + delta * np.random.normal(loc=0, scale=1, size=(1, 3))

        # grid of type 2 to type 1
        delta = delta * (0.5 ** (0.5 * hurst))
        for i in range(d, n - d + 1, D):
            x[i, 0] = f(delta, [x[i + d][0], x[i - d][0], x[i][d]])
            x[i, n] = f(delta, [x[i + d][n], x[i - d][n], x[i][n - d]])
            x[0, i] = f(delta, [x[0][i + d], x[0][i - d], x[d][i]])
            x[n, i] = f(delta, [x[n][i + d], x[n][i - d], x[n - d][i]])

        # interpolate and offset boundary points
        for i in range(d, n - d + 1, D):
            for j in range(D, n - d + 1, D):
                x[i][j] = f(delta, [x[i][j + d],
                                    x[i][j - d],
                                    x[i + d][j],
                                    x[i - d][j]])

        # interpolate and offset interior points
        for i in range(D, n - d + 1, D):
            for j in range(d, n - d + 1, D):  #
                x[i][j] = f(delta, [x[i][j + d],
                                    x[i][j - d],
                                    x[i + d][j],
                                    x[i - d][j]])

        if use_addition:
            for i in range(0, n + 1, D):
                for j in range(0, n + 1, D):
                    x[i][j] = x[i][j] + delta * np.random.normal(loc=0, scale=1, size=(1, 3))

            for i in range(d, n - d + 1, D):
                for j in range(d, n - d + 1, D):
                    x[i][j] = x[i][j] + delta * np.random.normal(loc=0, scale=1, size=(1, 3))
        D = int(D / 2)
        d = int(d / 2)


def fbm_2d(x, max_level, sigma, hurst, use_addition=False):
    # x: np.array of shape (n+1, n+1)
    # max_level so that N = 2^max_level
    # hurst so that D = 3 - H
    # boolean that enables or disables random addition correctino

    # generalize f3 and f4 for arbitrary sizes
    def f(delta_, x_):
        return np.mean(x_) + delta_ * np.random.normal(loc=0, scale=1)

    n = 2 ** (max_level)
    delta = sigma
    x[0, 0] = delta * np.random.normal(loc=0, scale=1)
    x[0, n] = delta * np.random.normal(loc=0, scale=1)
    x[n, 0] = delta * np.random.normal(loc=0, scale=1)
    x[n, n] = delta * np.random.normal(loc=0, scale=1)
    D = n  # prev_box_size
    d = int(n / 2)  # size of new box

    for _ in tqdm(range(1, max_level + 1)):
        # groung fro
        delta = delta * (0.5 ** (0.5 * hurst))
        for i in range(d, n - d + 1, D):
            for j in range(d, n - d + 1, D):
                x[i][j] = f(delta, [x[i + d][j + d],
                                    x[i + d][j - d],
                                    x[i - d][j + d],
                                    x[i - d][j - d]])

        if use_addition:
            for i in range(0, n + 1, D):
                for j in range(0, n + 1, D):
                    x[i][j] = x[i][j] + delta * np.random.normal(loc=0, scale=1)
        delta = delta * (0.5 ** (0.5 * hurst))

        for i in range(d, n - d + 1, D):
            x[i, 0] = f(delta, [x[i + d][0], x[i - d][0], x[i][d]])
            x[i, n] = f(delta, [x[i + d][n], x[i - d][n], x[i][n - d]])
            x[0, i] = f(delta, [x[0][i + d], x[0][i - d], x[d][i]])
            x[n, i] = f(delta, [x[n][i + d], x[n][i - d], x[n - d][i]])

        for i in range(d, n - d + 1, D):
            for j in range(D, n - d + 1, D):
                x[i][j] = f(delta, [x[i][j + d],
                                    x[i][j - d],
                                    x[i + d][j],
                                    x[i - d][j]])
        for i in range(D, n - d + 1, D):
            for j in range(d, n - d + 1, D):  #
                x[i][j] = f(delta, [x[i][j + d],
                                    x[i][j - d],
                                    x[i + d][j],
                                    x[i - d][j]])

        if use_addition:
            for i in range(0, n + 1, D):
                for j in range(0, n + 1, D):
                    x[i][j] = x[i][j] + delta * np.random.normal(loc=0, scale=1)

            for i in range(d, n - d + 1, D):
                for j in range(d, n - d + 1, D):
                    x[i][j] = x[i][j] + delta * np.random.normal(loc=0, scale=1)

        D = int(D / 2)
        d = int(d / 2)


def get_fbm_2d(hurst, max_level=7, sigma=1.):
    n = 2 ** max_level
    x = np.zeros([n + 1, n + 1])
    fbm_2d(x, max_level, sigma, hurst, True)
    return x
