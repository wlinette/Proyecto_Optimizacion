import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import solve_triangular

# ::::::::::::::::::Implementación del método Extended DWGM del paper ::::::::::::::::::
def extended_DWGM(f_func ,grad_func, hessian_func , ek_func, x0, Hk_gk_func = None, t = 1 ,max_iter = 10000, epsilon = 1e-6,delta = 0.9 ,gamma = 1e-4, m = 100):
    
    """Implementación del algoritmo Extended DWGM del paper 'An extended delayed weighted gradient algorithm for solving strongly convex optimization problems' de R.Andreani, H. Oviedo, M.Raydan y L.D. Secchin
    
    Recibe:

    f_func : función objetivo a minimizar
    grad_func : función que devuelve el gradiente de f_func
    hessian_func : función que devuelve la Hessiana de f_func
    ek_func : función que devuelve el valor de ek para cada iteración k
    x0 : punto inicial para la optimización
    Hk_gk_func : función que devuelve el producto Hk @ gk, si se proporciona se evita el cálculo explícito de la Hessiana (default = None)
    t : parámetro requerido para el algoritmo (default = 1)
    max_iter : número máximo de iteraciones para extended DWGM(default = 10000)
    epsilon : criterio de convergencia para extended DWGM (default = 1e-6)
    delta : parámetro requerido para el backtracking (default = 0.9)
    gamma : parámetro requerido para el backtracking (default = 1e-4)
    m : número máximo de iteraciones para el backtracking (default = 100)

    Retorna:
    history : diccionario con la información de cada iteración, incluyendo:
        - iter: número de iteración
        - x: punto actual de la iteración
        - f: valor de la función objetivo en el punto actual
        - g_norm: norma del gradiente en el punto actual
        - g_evals: número total de evaluaciones del gradiente realizadas
        - f_evals: número total de evaluaciones de la función objetivo realizadas
        - stop_reason: razón de parada (1 si se alcanzó la convergencia, 0 si se alcanzó el número máximo de iteraciones)
        - convergence_time: tiempo total de convergencia en segundos
    """
    x_k = x0.copy()
    g_k = grad_func(x_k)

    x_prev = x0.copy()
    g_prev = g_k.copy()
    
    gk_norm = np.linalg.norm(g_k)

    #listas para guardar la información de cada iteración
    history = {'iter': [], 'x': [], 'f': [], 'g_norm': [], 'g_evals': 1, 'f_evals': 1 }
    start_convergence_time=time.perf_counter()


    for k in range(max_iter):

        gk_norm_inf = np.linalg.norm(g_k, ord=np.inf)

        #Almacenamos la información de cada iteración
        history['iter'].append(k)
        history['x'].append(x_k.copy())
        history['f'].append(f_func(x_k))
        history['g_norm'].append(gk_norm_inf)
        

        if (gk_norm_inf <= epsilon):
            stop_reason = 1
            break

        # Si hay una función para calcular Hk @ gk, la usamos para evitar el cálculo explícito de la Hessiana, sino calculamos la Hessiana y luego el producto Hk @ gk
        if Hk_gk_func is not None:
            w_k = Hk_gk_func(grad_func,x_k, g_k)
        
        else: 
            H_k = hessian_func(x_k)
            #Paso 2 del algoritmo 1, página 3.
            w_k = H_k @ g_k
            
        gk_T_wk = np.dot(g_k,w_k)

        #Inicialización de alpha_k como ec. 2a (paso 3)
        alpha_k =  gk_T_wk / np.dot(w_k,w_k)
        #Paso 4
        z_k = x_k - t* alpha_k *g_k
        #Paso 5 
        r_k = grad_func(z_k)
        history['g_evals'] += 1
        # Backtracking (lineas 6-9 del algoritmo 1.)
        rk_norm = np.linalg.norm(r_k)

        for i in range(m):
            if (rk_norm**2 <= (gk_norm**2 - (gamma*t*alpha_k) * gk_T_wk)):
                break
            alpha_k = delta * alpha_k
            z_k = x_k - (t * alpha_k * g_k)
            r_k = grad_func(z_k)
            history['g_evals'] += 1
            rk_norm = np.linalg.norm(r_k)
            
        # Calculo de beta_k, lineas 10-11 del algoritmo 1.
        y_k = r_k - g_prev
        beta_k = - np.dot(g_prev,y_k)/np.dot(y_k,y_k)

        # actualización de x_{k+1} con paso retrasado, linea 12 del algoritmo 1.
        x_next = x_prev + beta_k*(z_k - x_prev)
        g_next = grad_func(x_next)
        history['g_evals'] += 1

        g_next_norm = np.linalg.norm(g_next)
        cond2 = rk_norm**2 + min (ek_func(k, gamma, t, alpha_k, gk_T_wk), gamma * t * alpha_k * gk_T_wk)
        if  g_next_norm**2 > cond2:
            x_next = z_k
            g_next = r_k
            g_next_norm = np.linalg.norm(g_next)
        
        # Actualización de variables para la siguiente iteración
        x_prev, x_k = x_k, x_next
        g_prev, g_k = g_k, g_next
        gk_norm = g_next_norm

    else:
        stop_reason = 0
    
    history['stop_reason'] = stop_reason
    end_convergence_time = time.perf_counter()
    history['convergence_time'] = end_convergence_time - start_convergence_time
    return history

# :::::::::::::::::::::::::::: Función para obtener el minimo con Gradiente Conjugado de SciPy :::::::::::::::::::::::::::::::::
def cg_scipy(f_func, grad_func, x0, callback, args = None , tol=1e-8, maxiter=10000):
    """Función para obtener el mínimo de una función utilizando el método de Gradiente Conjugado de SciPy.
    Recibe:
    - f_func : función objetivo a minimizar
    - grad_func : función que devuelve el gradiente de f_func
    - x0 : punto inicial para la optimización
    - callback : función que se llama después de cada iteración (default = None)
    - args : tupla de argumentos adicionales para f_func y grad_func (default = None)
    - tol : criterio de convergencia para el método CG (default = 1e-8)
    - maxiter : número máximo de iteraciones para el método CG (default = 10000)

    Retorna:
    - res : resultado de la optimización con el método CG de SciPy
    """

    start_convergence_time=time.perf_counter()

    res = minimize(
        fun=f_func,
        x0=x0,
        method='CG',
        args = args if args is not None else (),
        jac=grad_func,
        tol=tol,
        options={'maxiter': maxiter, 'gtol': tol},
        callback=callback
    )
    end_convergence_time = time.perf_counter()
    convergence_time = end_convergence_time - start_convergence_time

    return res, convergence_time

def run_cg(f_func, grad_func, x0, args = None, tol = 1e-8, maxiter = 10000):

    """Función para ejecutar el método de Gradiente Conjugado de SciPy y almacenar la información de cada iteración en un diccionario.
    Recibe:
    - f_func : función objetivo a minimizar
    - grad_func : función que devuelve el gradiente de f_func
    - x0 : punto inicial para la optimización
    - args : tupla de argumentos adicionales para f_func y grad_func (default = None)
    - tol : criterio de convergencia para el método CG (default = 1e-8)
    - maxiter : número máximo de iteraciones para el método CG (default = 10000)
     Retorna:
     - history : diccionario con la información de cada iteración, incluyendo: 
        - iter: número de iteración
        - x: punto actual de la iteración
        - f: valor de la función objetivo en el punto actual
        - g_norm: norma del gradiente en el punto actual
        - g_evals: número total de evaluaciones del gradiente realizadas
        - f_evals: número total de evaluaciones de la función objetivo realizadas
        - stop_reason: razón de parada (1 si se alcanzó la convergencia, 0 si se alcanzó el número máximo de iteraciones)
        - convergence_time: tiempo total de convergencia en segundos"""

    history = {'iter': [], 'x': [],'f': [], 'g_norm': []}

    def my_callback(xk):
        # Calculamos los valores en el punto actual xk
        f_val = f_func(xk)
        grad_val = grad_func(xk)
        g_norm = np.linalg.norm(grad_val, ord=np.inf)

        # Guardamos en el historial
        history['iter'].append(len(history['iter']))
        history['x'].append(xk.copy())
        history['f'].append(f_val)
        history['g_norm'].append(g_norm)

    res, convergence_time = cg_scipy(f_func, grad_func, x0, my_callback, args=args, tol=tol, maxiter=maxiter)
    history['convergence_time'] = convergence_time
    history['f_evals'] = res.nfev
    history['g_evals'] = res.njev
    history['stop_reason'] = 1 if res.success else 0
    return history

# ::::::::::::::::: Implementación de gradiente descendente con backtracking :::::::::::::::::

"""
        Funcion para obtener el tamanio de paso con backtracking Armijo

Parametros:
    -f(): Funcion de costo
    -grad_f(): Funcion para obtener el gradiente de la funcion de costo
    -xk: Punto inicial
    -alpha: Tamaño de paso inicial
    -rho: Hiperparámetro para backtracking de Armijo, p in (0,1)
    -c1: Hiperparámetro para backtracking de Armijo, c1 in (0,1)
    -max_it: Iteraciones máximas para backtracking

"""
def armijo_backtrack(f,grad_f,z,f_z,g_z,alpha=1,rho=0.5,c1=1e-4,max_it=100):

    #Obtencion de f(z,y,X) y gradiente(z,y,X)
    d_z=-g_z

    #f(zk+ alpha*rho_k)
    f_z_adk=f(z+ alpha*d_z)
    iter=0
    
    #Terminos necesarios para la condicion de aceptacion
    term=c1*(np.dot(g_z,d_z))

    #Ejecutar mientras no se cumpla la condicion de Armijo o mientras no se alcance el maximo de iteraciones
    while (( f_z_adk > f_z + alpha*term ) and iter <max_it):
        #Obtencion de alpha
        alpha=rho*alpha

        #Actualizacion de terminos para la condcion de aceptacion
        f_z_adk=f(z+ alpha*d_z)
        iter+=1
        
    return alpha,iter



"""
        Funcion para obtener gradiente descendente con Armijo

Parametros de la funcion:
    -f(): Funcion de costo
    -grad_f(): Funcion para obtener el gradiente de la funcion de costo
    -xk: Punto inicial
    -alpha: Tamaño de paso inicial
    -rho: Hiperparámetro para backtracking de Armijo, p in (0,1)
    -c1: Hiperparámetro para backtracking de Armijo, c1 in (0,1)
    -max_it_GD: Iteraciones máximas para el descenso de gradiente
    -tol_GD: Tolerancia para paro de descenso de gradiente
    -max_it_armijo: Iteraciones maximas para backtracking de Armijo

Retorna:
    - history : diccionario con la información de cada iteración, incluyendo:
        - iter: número de iteración
        - x: punto actual de la iteración
        - f: valor de la función objetivo en el punto actual
        - g_norm: norma del gradiente en el punto actual
        - g_evals: número total de evaluaciones del gradiente realizadas
        - f_evals: número total de evaluaciones de la función objetivo realizadas
        - stop_reason: razón de parada (1 si se alcanzó la convergencia, 0 si se alcanzó el número máximo de iteraciones)
        - convergence_time: tiempo total de convergencia en segundos

"""
#
def desc_grad_armijo(f,grad_f,z0, alpha_0=1.0,rho=0.5,c1=1e-4,max_it_GD=10000,tol_GD=1e-6, max_it_Armijo=100):

    z=z0.copy()

    #Gradiente f(z,y,X) 
    g_z=grad_f(z)

    #Contador para las iteraciones
    k=0

    #diccionario para guardar info por iteracion
    history = {'iter': [], 'x': [], 'f': [], 'g_norm': [], 'g_evals': 1, 'f_evals': 1 }

    #Inicio de cronometro
    start_convergence_time=time.perf_counter()

    while (np.linalg.norm(g_z, ord = np.inf) >= tol_GD) and (k < max_it_GD):


        fz_actual=f(z)
        history['f_evals'] += 1
        norm_g=np.linalg.norm(g_z, ord = np.inf)

        #Adicion de informacion a las listas
        history['iter'].append(k)
        history['x'].append(z.copy())
        history['f'].append(fz_actual)
        history['g_norm'].append(norm_g)
         
        #Direccion de descenso
        pk=-g_z 
        #Busqueda de paso con backtracking
        alpha,iter=armijo_backtrack(f,grad_f,z,fz_actual,g_z,alpha_0,rho,c1,max_it_Armijo)


        #Actualizacion
        z=z+alpha*pk 
        #Actualizar gradiente e iteracion actual
        g_z=grad_f(z)
        history['g_evals'] += 1
        history['f_evals'] += (iter+1) #iteraciones de backtracking + evaluacion de f en z+alpha*pk
        k+=1

        #Finalizacion de cronometro por iteracion
        end_iter_time=time.perf_counter()

    
    #Finalizacion cronometro
    end_convergence_time=time.perf_counter()
    dur_convergence_time=end_convergence_time-start_convergence_time

    if (k<max_it_GD):
        stop_reason=1

    else :
        stop_reason=0

    history['stop_reason'] = stop_reason
    history['convergence_time'] = dur_convergence_time
    
    return history



#::::::::::::::: Implementación del método trust region dogleg ::::::::::::::::::


"""Funcion para calcular el punto de Cauchy

-g_k : gradiente(z_k)
-B_k: hessiano(z_k)
-r_k: radio de confianza en iteracion k

"""
def Cauchy_point(g_k,B_k,r_k):

    tau=0.0
    g_B_g=np.dot(g_k, B_k @ g_k)
    
    #Obtencion del tau que minimiza mk sujeto a || tau * pk|| < r_k
    if g_B_g<=0:
        tau=1
    
    else:
        tau=min(1, np.linalg.norm(g_k)**3 / (r_k * g_B_g))

    #calculo del punto de Cauchy
    pk_c= - (tau* r_k/ np.linalg.norm(g_k))* g_k

    return pk_c 
    


"""
Funcion para resolver una ecuacion cuadratica de la forma ax^2+ bx +c=0
"""
def solve_cuadratic(a,b,c):

    d=max(0.0,b**2 - (4*a*c))
    s1=( - b + np.sqrt(d))/(2*a)
    s2=( - b - np.sqrt(d))/(2*a)

    return s1,s2


"""
Metodo dogleg

-g_k : gradiente(z_k)
-B_k: hessiano(z_k)
-r_k: radio de confianza en iteracion k

"""
def dogleg(g_k,B_k,r_k):

    g_B_g=np.dot(g_k, B_k @ g_k)
    
    #Paso de maximo descenso
    pk_u= -( np.dot(g_k,g_k)/ g_B_g ) * g_k

    #Verificar que B_k sea definida positiva
    try:
        L=np.linalg.cholesky(B_k)
    
    except np.linalg.LinAlgError:
        #print("La matriz $B_k$ no es definida positiva. Devolviendo punto de Cauchy")
        return Cauchy_point(g_k,B_k,r_k)

    #Paso de Newton pk_B= -Bk_inv * g_k, para no sacar la inversa resolvemos sistemas de ecuaciones

    #Resolvemos Ly=gk
    y=solve_triangular(L,-g_k,lower=True)
    #Resolvemos L^T pk_B=y
    pk_B=solve_triangular(L.T,y,lower=False)


    #Obtener pk
    if (np.linalg.norm(pk_B)<=r_k):
        return pk_B
    
    else:

        if (np.linalg.norm(pk_u)>=r_k):
            return Cauchy_point(g_k,B_k,r_k)
        
        else:
            #Resolver el sistema al^2+ bl+c=0
            a= np.linalg.norm(pk_B-pk_u)**2
            b= 2*np.dot(pk_u, pk_B-pk_u)#2* pk_B.T @ (pk_B-pk_u) #verificar si es pk_U en vez de pk_B
            c= np.linalg.norm(pk_u)**2 - r_k**2

            l1,l2=solve_cuadratic(a,b,c)

            tau=max(l1,l2)+1
            
            if 0<=tau<=1:
                return tau * pk_u

            if 1<=tau<=2:
                return pk_u+ (tau-1)*(pk_B-pk_u)

    return -1
            

"""
Metodo de region de confianza

- f_func: funcion para evaluar
- g_func: funcion para obtener el gradiente de f
- H_func: funcion para obtener el hessiano de f
- z0: punto inicial
- y: vector de etiquetas, yi=0 si la imagen corresponde a un 0, yi=1 si la imagen corresponde a 1
- X: matriz donde cada fila es una imagen, la fila final es de 1s.
- r_hat: limite del radio de confianza minimo
- r_0: radio de confianza inicial
- eta: parametro del metodo de region de confianza
- max_it: iteraciones maximas
- tol: tolerancia minima para el gradiente
"""
def trust_region_dogleg(f_func,g_func,H_func,z0,r_hat=1.0,r_0=0.1,eta=0.33, max_it=300, tol=1e-8):
    
    z_k=np.array(z0,dtype=float)
    r_k=r_0
    
    history = {'iter': [], 'x': [], 'f': [], 'g_norm': [], 'g_evals': 0, 'f_evals': 0 }
    start_time = time.perf_counter()

    for k in range(max_it):
        
        #Obtencion de f, gradiente,||gradiente|| y hessiano
        f_k=f_func(z_k)
        g_k=g_func(z_k)
        B_k=H_func(z_k)
        history['f_evals'] += 1
        history['g_evals'] += 1
        norm_g=np.linalg.norm(g_k, ord = np.inf)

        #guardar la informacion necesaria para las graficas
        history['iter'].append(k)
        history['x'].append(z_k.copy())
        history['f'].append(f_k)
        history['g_norm'].append(norm_g)

        #condicion de paro por gradiente pequeno
        if norm_g<tol:
            stop_reason = 1
            break

        #obtencion del tamanio de paso con dogleg
        pk=dogleg(g_k,B_k,r_k)

        f_k_pk=f_func(z_k+pk)
        history['f_evals'] += 1

        curr_reduction= f_k -f_k_pk
        pred_reduction= - (np.dot(g_k,pk) + 0.5* np.dot(pk,B_k @ pk))

        #calculo de rho_k
        if pred_reduction<=0:
            rho_k=0
        else:
            rho_k=curr_reduction/ pred_reduction

        # actualizacion del radio de confianza en funcion del valor de rho_k
        if rho_k< 0.25:
            r_k=0.25*r_k
        
        else:
            if rho_k>0.75 and np.isclose(np.linalg.norm(pk),r_k):
                r_k=min(2*r_k,r_hat)


        if rho_k> eta:
            z_k=z_k+pk

    else:
        stop_reason = 0

    total_time=time.perf_counter() - start_time
    history['stop_reason'] = stop_reason
    history['convergence_time'] = total_time

    return history

