import numpy as np

# :::::::::::::::::: FUNCION AUXILIAR PARA EL ALGORITMO EXTENDED DWGM :::::::::::::::::
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


#:::::::::::::::::: FUNCIONES EXPERIMENTO 1 Y 2 DEL PAPER :::::::::::::::::

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




#:::::::::::::::::: FUNCIONES EXPERIMENTO 3 DEL PAPER :::::::::::::::::

def f2_func(x, lambda_val = None):
    """Función f2(x) = - log (lambda^2 - x^T x) (experimento 3 del paper)
    Recibe:
    - x : una lista 
    - lambda_val : un escalar positivo

    Retorna:
    - f2(x) : numpy array
    """
    if lambda_val is None:
        lambda_val = (10.0 * len(x))**0.5  # Valor predeterminado de lambda usado en paper

    x = np.array(x, dtype= np.float64)
    dot_p = np.dot(x, x)
    output = - np.log(lambda_val**2 - dot_p)

    return output


def grad2_func(x, lambda_val = None):
    """Gradiente de f2(x) = - log (lambda^2 - x^T x) (experimento 3 del paper)
    Recibe:
    - x : una lista 
    - lambda_val : un escalar positivo

    Retorna:
    - grad_f2(x) : numpy array
    """
    if lambda_val is None:
        lambda_val = (10.0 * len(x))**0.5  # Valor predeterminado de lambda usado en paper

    x = np.array(x, dtype= np.float64)
    dot_p = np.dot(x, x)
    output = (2*x) / (lambda_val**2 - dot_p)

    return output

def hessian2_func(x, lambda_val = None):
    """Hessiana de f2(x) = - log (lambda^2 - x^T x) (experimento 3 del paper)
    Recibe:
    - x : una lista 
    - lambda_val : un escalar positivo

    Retorna:
    - hess_f2(x) : numpy array
    """
    if lambda_val is None:
        lambda_val = (10.0 * len(x))**0.5  # Valor predeterminado de lambda usado en paper

    x = np.array(x, dtype= np.float64)
    dot_p = np.dot(x, x)
    n = len(x)
    
    # Matriz identidad
    I = np.eye(n)
    
    # Cálculo de la Hessiana
    term1 = (2 / (lambda_val**2 - dot_p)) * I
    term2 = (4 / (lambda_val**2 - dot_p)**2) * np.outer(x, x)
    
    output = term1 + term2
    
    return output


# :::::::::::::::::: FUNCIONES EXPERIMENTO 4 DEL PAPER :::::::::::::::::
def f3_func(x, z, y, sigma):
    """ 
    Función experimento 4,f(x) = (sigma/2)*||x||^2 + sum(log(1 + e^{-(x^T z^i)y^i}))
    """
    x = np.array(x, dtype=np.float64) #n
    z = np.array(z, dtype=np.float64) # mxn
    y = np.array(y, dtype=np.float64) # m x c
    

    dot_products = z @ x 
    loss_vector = np.logaddexp(0, -dot_products * y)
    
    term1 = (sigma / 2) * np.linalg.norm(x)**2
    term2 = np.sum(loss_vector)
    
    return term1 + term2

def grad3_func(x, z, y, sigma):
    """ 
    Gradiente de la función del experimento 4
    """
    x = np.array(x, dtype=np.float64) 
    z = np.array(z, dtype=np.float64) 
    y = np.array(y, dtype=np.float64) 

    # (x^T z^i) * y^i 
    v = (z @ x) * y
    
    weights = y / (1 + np.exp(v))
    
    # sum(weights_i * z^i) 
    gradient = sigma * x - (z.T @ weights)
    
    return gradient


def hessian3_func(x, z, y, sigma):
    """
    Hessiano para la función del experimento 4.
    Retorna una matriz de n x n.
    """
    m, n = z.shape
    v = (z @ x) * y
    
    # h_i / (1 + h_i)^2 es equivalente a sigmoid(-v) * sigmoid(v)
    # o de forma más simple: p * (1 - p) donde p = 1 / (1 + exp(v))
    p = 1.0 / (1.0 + np.exp(v))
    weights = p * (1.0 - p) # Vector de dimensión m
    
    # Forma matricial: sigma * I + Z.T @ diag(weights) @ Z
    # Esto evita el bucle for sobre m observaciones.
    I = np.eye(n)
    # np.diag(weights) @ z multiplica cada fila i de z por weights[i]
    hessian = sigma * I + (z.T * weights) @ z
    
    return hessian

def get_hessian_vector_product(grad_func, x_k, g_k):
    """
    Versión simplificada que solo recibe lo necesario en la iteración k.
    Implementa el h dinámico de la página 9[cite: 1726, 1727].
    """
    gk_norm = np.linalg.norm(g_k)
    h = 1e-5 / min(1.0, max(1e-3, 1e5 * gk_norm)) # [cite: 1726, 1727]
    
    # grad_func ya debe venir con z, y, sigma pre-configurados
    g_plus = grad_func(x_k + h * g_k) 
    wk = (g_plus - g_k) / h # [cite: 1722]
    
    return wk