import numpy as np

def ek_func(k, gamma, t, alpha_k, gk_T_wk):
    """Función que devuelve el resultado de ek
    Recibe:
    - k : número de iteración
    - gamma : parámetro usado para el backtracking
    - t : parámetro requerido para el algoritmo de extended DWGM
    - alpha_k : paso de la iteración k
    - gk_T_wk : producto punto de gk y wk
    Retorna:
    - ek : escalar
    """

    if k == 0:
        k_pow2 = 1
    else:
        k_pow2 = k**2
    
    ek = min(1/k_pow2,0.9 * gamma * t * alpha_k * gk_T_wk)

    return ek


def f_func(x):
    """Función f(x) = sum _{i=1}^{n} [i/10 * (e^{x_i} - x_i)] (experimento 1 del paper)
    Recibe:
    - x : una lista 

    Retorna:
    - f(x) : numpy array
    """
    x = np.array(x, dtype= np.float64)
    n = len(x)
    idx_arr = (np.arange(1, n+1) / 10.0)
    output = np.sum((idx_arr)*(np.exp(x) - x))
    return output


def grad_func(x):
    """Gradiente de f(x) = sum _{i=1}^{n} [i/10 * (e^{x_i} - x_i)] (experimento 1 del paper)
    Recibe:
    - x : una lista 
    Retorna:
    - grad_f(x) : numpy array"""

    x = np.array(x, dtype= np.float64)
    n = len(x)
    idx_arr = (np.arange(1, n+1) / 10.0)
    output = (idx_arr)*(np.exp(x) - 1)
    return output


def hessian_func(x):
    """Hessiana de f(x) = sum _{i=1}^{n} [i/10 * (e^{x_i} - x_i)] (experimento 1 del paper)
    Recibe:
    - x : una lista 
    Retorna:
    - hess_f(x) : numpy array"""

    x = np.array(x, dtype= np.float64)
    n = len(x)
    idx_arr = (np.arange(1, n+1) / 10.0)
    output = idx_arr * np.exp(x)
    return np.diag(output)


def Hk_gk_func(x):
    """Función que devuelve el resultado de Hk*gk
    Recibe:
    - Hk : numpy array con información de la Hessiana
    - gk : numpy array con el gradiente de la función en el punto xk
    Retorna:
    - Hk*gk : numpy array
    """
    Hk = hessian_func(x)
    gk = grad_func(x)
    return Hk*gk