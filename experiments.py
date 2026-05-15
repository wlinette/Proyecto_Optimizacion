import method_functions as meth_f
import loss_functions as loss_f
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def plot_convergence(histories,method_names, path = None):
    """ Función para graficar la convergencia de los métodos. Recibe:
    - histories: lista de diccionarios con la historia de cada método
    - method_names: lista de nombres de los métodos para la leyenda
    """
    plt.figure(figsize=(12, 6))
    
    for history, name in zip(histories, method_names):
        plt.plot(history['g_norm'], label=name)
    
    plt.yscale('log')
    plt.xlabel('Iteración')
    plt.ylabel(r'$||g_k||_{\infty}$')
    plt.title(r'$|| \nabla f(x_k)||_{\infty}$ a lo largo de las iteraciones')
    plt.legend()
    plt.grid()

    if path is not None:
        plt.savefig(path)

    plt.pause(8)
    plt.close()


def sigmoid(z):
    #limitar los valores de z para evitar errores de overflow
    z = np.clip(z, -250, 250)
    return 1.0 / (1.0 + np.exp(-z))

def predict(Z, X):
    #obtención de la probabilidad
    predictions = sigmoid(Z @ X)
    #clasificación binaria con umbral 0.5
    predictions = np.where(predictions >= 0.5, 1, -1)
    return predictions

# Configurar para mostrar todas las columnas y filas de los dataframes sin truncar
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# :::::::::::::::::: EXPERIMENTO 1 DEL PAPER :::::::::::::::::

epsilon = 1e-8
n = 1000 
x0 = np.array([2.0]*n, dtype=np.float64)
saving_path = 'experiments/exp1.png' 
print(r"Experimento 1 : $f_1(x) = \sum _{i=1} ^n \frac{i}{10} (e^{x_i} - x_i)$")

"""EXTENDED DWGM """
history_extDWGM= meth_f.extended_DWGM(loss_f.f_func ,loss_f.grad_func, loss_f.hessian_func , loss_f.ek_func , x0, t = 1 ,max_iter = 300, epsilon = epsilon,delta = 0.9 ,gamma = 1e-4, m = 100)


"""GRADIENTE CONJUGADO DE SCIPY"""
history_CG = meth_f.run_cg(loss_f.f_func, loss_f.grad_func, x0, tol=epsilon, maxiter=300)


""" GRADIENTE DESCENDENTE CON BACKTRACKING """
history_GD = meth_f.desc_grad_armijo(loss_f.f_func, loss_f.grad_func, x0, alpha_0=1.0, rho=0.5, c1=1e-4, max_it_GD=300, tol_GD=epsilon, max_it_Armijo=100)


""" TRUST REGION """
history_TR = meth_f.trust_region_dogleg(loss_f.f_func, loss_f.grad_func, loss_f.hessian_func,x0, r_hat=1.0, r_0=0.1, eta=0.33, max_it=300, tol=epsilon)


resume_dic = {'Method': ['Ext . DWGM', 'CG', 'G. Descent', 'Trust Region'],
              'Iter': [history_extDWGM['iter'][-1], history_CG['iter'][-1], history_GD['iter'][-1], history_TR['iter'][-1]],
              'Grad. Evals.': [history_extDWGM['g_evals'], history_CG['g_evals'], history_GD['g_evals'], history_TR['g_evals']],
              'F Evals.': [history_extDWGM['f_evals'], history_CG['f_evals'], history_GD['f_evals'], history_TR['f_evals']],
              'Best G Norm': [history_extDWGM['g_norm'][-1], history_CG['g_norm'][-1], history_GD['g_norm'][-1], history_TR['g_norm'][-1]],
              'Best F Value': [history_extDWGM['f'][-1], history_CG['f'][-1], history_GD['f'][-1], history_TR['f'][-1]],
              'Stop reason': [history_extDWGM['stop_reason'], history_CG['stop_reason'], history_GD['stop_reason'], history_TR['stop_reason']]}

df_summary = pd.DataFrame(resume_dic)
print(df_summary)
print('\n')
plot_convergence([history_extDWGM, history_CG, history_GD, history_TR], ['DWGM modificado', 'Gradiente Conjugado', 'Gradiente Descendente', 'Región de Confianza'], path=saving_path)




# # :::::::::::::::::: EXPERIMENTO 2 DEL PAPER (misma función f1, distinto x0) :::::::::::::::::
np.random.seed(32)
x0 = np.array(np.random.uniform(-2, 2, size=(n)))
saving_path = 'experiments/exp2.png'
print(r"Experimento 2 : $f_1(x) = \sum _{i=1} ^n \frac{i}{10} (e^{x_i} - x_i)$")

"""EXTENDED DWGM """
history_extDWGM= meth_f.extended_DWGM(loss_f.f_func ,loss_f.grad_func, loss_f.hessian_func , loss_f.ek_func , x0, t = 1 ,max_iter = 300, epsilon = epsilon,delta = 0.9 ,gamma = 1e-4, m = 100)


"""GRADIENTE CONJUGADO DE SCIPY"""
history_CG = meth_f.run_cg(loss_f.f_func, loss_f.grad_func, x0, tol=epsilon, maxiter=300)


""" GRADIENTE DESCENDENTE CON BACKTRACKING """
history_GD = meth_f.desc_grad_armijo(loss_f.f_func, loss_f.grad_func, x0, alpha_0=1.0, rho=0.5, c1=1e-4, max_it_GD=300, tol_GD=epsilon, max_it_Armijo=100)


""" TRUST REGION """
history_TR = meth_f.trust_region_dogleg(loss_f.f_func, loss_f.grad_func, loss_f.hessian_func,x0, r_hat=1.0, r_0=0.1, eta=0.33, max_it=300, tol=epsilon)


resume_dic = {'Method': ['Ext . DWGM', 'CG', 'G. Descent', 'Trust Region'],
              'Iter': [history_extDWGM['iter'][-1], history_CG['iter'][-1], history_GD['iter'][-1], history_TR['iter'][-1]],
              'Grad. Evals.': [history_extDWGM['g_evals'], history_CG['g_evals'], history_GD['g_evals'], history_TR['g_evals']],
              'F Evals.': [history_extDWGM['f_evals'], history_CG['f_evals'], history_GD['f_evals'], history_TR['f_evals']],
              'Best G Norm': [history_extDWGM['g_norm'][-1], history_CG['g_norm'][-1], history_GD['g_norm'][-1], history_TR['g_norm'][-1]],
              'Best F Value': [history_extDWGM['f'][-1], history_CG['f'][-1], history_GD['f'][-1], history_TR['f'][-1]],
              'Stop reason': [history_extDWGM['stop_reason'], history_CG['stop_reason'], history_GD['stop_reason'], history_TR['stop_reason']]}

df_summary = pd.DataFrame(resume_dic)
print(df_summary)
print('\n')

plot_convergence([history_extDWGM, history_CG, history_GD, history_TR], ['DWGM modificado', 'Gradiente Conjugado', 'Gradiente Descendente', 'Región de Confianza'], path=saving_path)


# :::::::::::::::::: EXPERIMENTO 3 DEL PAPER :::::::::::::::::
x0 = np.array([2.0]*n, dtype=np.float64)
saving_path = 'experiments/exp3.png'

print(r"Experimento 3 : $ f_2(x) - log (\lambda^2 - x^T x)$")

"""EXTENDED DWGM """
history_extDWGM= meth_f.extended_DWGM(loss_f.f2_func ,loss_f.grad2_func, loss_f.hessian2_func , loss_f.ek_func , x0, t = 1 ,max_iter = 300, epsilon = epsilon,delta = 0.9 ,gamma = 1e-4, m = 100)


"""GRADIENTE CONJUGADO DE SCIPY"""
history_CG = meth_f.run_cg(loss_f.f2_func, loss_f.grad2_func, x0, tol=epsilon, maxiter=300)


""" GRADIENTE DESCENDENTE CON BACKTRACKING """
history_GD = meth_f.desc_grad_armijo(loss_f.f2_func, loss_f.grad2_func, x0, alpha_0=1.0, rho=0.5, c1=1e-4, max_it_GD=300, tol_GD=epsilon, max_it_Armijo=100)


""" TRUST REGION """
history_TR = meth_f.trust_region_dogleg(loss_f.f2_func, loss_f.grad2_func, loss_f.hessian2_func,x0, r_hat=1.0, r_0=0.1, eta=0.33, max_it=300, tol=epsilon)


resume_dic = {'Method': ['Ext . DWGM', 'CG', 'G. Descent', 'Trust Region'],
              'Iter': [history_extDWGM['iter'][-1], history_CG['iter'][-1], history_GD['iter'][-1], history_TR['iter'][-1]],
              'Grad. Evals.': [history_extDWGM['g_evals'], history_CG['g_evals'], history_GD['g_evals'], history_TR['g_evals']],
              'F Evals.': [history_extDWGM['f_evals'], history_CG['f_evals'], history_GD['f_evals'], history_TR['f_evals']],
              'Best G Norm': [history_extDWGM['g_norm'][-1], history_CG['g_norm'][-1], history_GD['g_norm'][-1], history_TR['g_norm'][-1]],
              'Best F Value': [history_extDWGM['f'][-1], history_CG['f'][-1], history_GD['f'][-1], history_TR['f'][-1]],
              'Stop reason': [history_extDWGM['stop_reason'], history_CG['stop_reason'], history_GD['stop_reason'], history_TR['stop_reason']]}

df_summary = pd.DataFrame(resume_dic)
print(df_summary)
print('\n')

plot_convergence([history_extDWGM, history_CG, history_GD, history_TR], ['DWGM modificado', 'Gradiente Conjugado', 'Gradiente Descendente con Backtracking', 'Región de Confianza'], path=saving_path)


# :::::::::::::::::: EXPERIMENTO 4 DEL PAPER :::::::::::::::::::::::

# importacion de datos de ionosphere
ionosphere = fetch_ucirepo(id=52) 
Z = ionosphere.data.features # matriz de caracteristicas, normalizada entre -1 y 1
y = ionosphere.data.targets  # etiquetas g: good, b: bad

#conversión de etiquetas a formato numérico y convesión de dataframe a numpy array
Z = Z.to_numpy()
y_mapped = y.iloc[:, 0].map({'g': 1, 'b': -1})
y = y_mapped.to_numpy()

n = Z.shape[1]  # Número de características
x0 = np.array([1.0]*n, dtype=np.float64)
sigma = 0.0 # parametro requerido para la funcion f3
saving_path = 'experiments/exp4.png'
print(r"Experimento 4 : $ f_3(x)= \frac{\sigma}{2} || x|| ^2 + \sum log (1 + e^{-(x^Tz^i)y^i})$")

history_extDWGM = meth_f.extended_DWGM(lambda x: loss_f.f3_func(x, Z, y,sigma), lambda x: loss_f.grad3_func(x, Z, y, sigma), lambda x: loss_f.hessian3_func(x, Z, y, sigma), loss_f.ek_func, x0,lambda grad3_func,x, gk: loss_f.get_hessian_vector_product(grad3_func, x, gk), t = 1, max_iter = 300, epsilon = epsilon, delta = 0.9, gamma = 1e-4, m = 100)



"""GRADIENTE CONJUGADO DE SCIPY"""
history_CG = meth_f.run_cg(lambda x: loss_f.f3_func(x, Z, y,sigma), lambda x: loss_f.grad3_func(x, Z, y, sigma), x0, tol=epsilon, maxiter=300)


""" GRADIENTE DESCENDENTE CON BACKTRACKING """
history_GD = meth_f.desc_grad_armijo(lambda x: loss_f.f3_func(x, Z, y, sigma), lambda x: loss_f.grad3_func(x, Z, y, sigma), x0, alpha_0=1.0, rho=0.5, c1=1e-4, max_it_GD=300, tol_GD=epsilon, max_it_Armijo=100)


""" TRUST REGION """
history_TR = meth_f.trust_region_dogleg(lambda x: loss_f.f3_func(x, Z, y,sigma), lambda x: loss_f.grad3_func(x, Z, y, sigma), lambda x: loss_f.hessian3_func(x, Z, y, sigma),x0, r_hat=1.0, r_0=0.1, eta=0.33, max_it=300, tol=epsilon)


resume_dic = {'Method': ['Ext . DWGM', 'CG', 'G. Descent', 'Trust Region'],
              'Iter': [history_extDWGM['iter'][-1], history_CG['iter'][-1], history_GD['iter'][-1], history_TR['iter'][-1]],
              'Grad. Evals.': [history_extDWGM['g_evals'], history_CG['g_evals'], history_GD['g_evals'], history_TR['g_evals']],
              'F Evals.': [history_extDWGM['f_evals'], history_CG['f_evals'], history_GD['f_evals'], history_TR['f_evals']],
              'Best G Norm': [history_extDWGM['g_norm'][-1], history_CG['g_norm'][-1], history_GD['g_norm'][-1], history_TR['g_norm'][-1]],
              'Best F Value': [history_extDWGM['f'][-1], history_CG['f'][-1], history_GD['f'][-1], history_TR['f'][-1]],
              'Stop reason': [history_extDWGM['stop_reason'], history_CG['stop_reason'], history_GD['stop_reason'], history_TR['stop_reason']]}

df_summary = pd.DataFrame(resume_dic)
print(df_summary)
print('\n')
plot_convergence([history_extDWGM, history_CG, history_GD, history_TR], ['DWGM modificado', 'Gradiente Conjugado', 'Gradiente Descendente con Backtracking', 'Región de Confianza'], path=saving_path)



# :::::::::::::::::::::::: EXPERIMENTO 5 : CLASIFICADOR LOGÍSTICO CON REGULARIZACIÓN L2 :::::::::::::::::::::::::

# División del conjunto de datos (Train / Test Split), (20% para prueba) y stratify=y para balancear las clases
Z_train, Z_test, y_train, y_test = train_test_split(
    Z, y, test_size=0.2, random_state=2, stratify=y
)

n = Z_train.shape[1]  # Número de características
x0 = np.array([1.0]*n, dtype=np.float64)

history_extDWGM = meth_f.extended_DWGM(lambda x: loss_f.f3_func(x, Z_train, y_train,sigma), lambda x: loss_f.grad3_func(x, Z_train, y_train, sigma), lambda x: loss_f.hessian3_func(x, Z_train, y_train, sigma), loss_f.ek_func, x0,lambda grad3_func,x, gk: loss_f.get_hessian_vector_product(grad3_func, x, gk), t = 1, max_iter = 300, epsilon = epsilon, delta = 0.9, gamma = 1e-4, m = 100)
history_GD = meth_f.desc_grad_armijo(lambda x: loss_f.f3_func(x, Z_train, y_train, sigma), lambda x: loss_f.grad3_func(x, Z_train, y_train, sigma), x0, alpha_0=1.0, rho=0.5, c1=1e-4, max_it_GD=300, tol_GD=epsilon, max_it_Armijo=100)


# Hacemos predicciones con los pesos obtenidos por cada método
y_pred_extDWGM = predict(Z_test, history_extDWGM['x'][-1])
y_pred_GD = predict(Z_test, history_GD['x'][-1])

# Evaluamos
test_accuracy_extDWGM = accuracy_score(y_test, y_pred_extDWGM)
train_accuracy_extDWGM = accuracy_score(y_train, predict(Z_train, history_extDWGM['x'][-1]))
test_accuracy_GD = accuracy_score(y_test, y_pred_GD)
train_accuracy_GD = accuracy_score(y_train, predict(Z_train, history_GD['x'][-1]))

print("Resultados del Experimento 5: Clasificador Logístico con Regularización L2")
print(f"Train accuracy with Extended DWGM: {train_accuracy_extDWGM * 100:.2f}%")
print(f"Train accuracy with Gradient Descent: {train_accuracy_GD * 100:.2f}%")
print(f"Test accuracy with Extended DWGM: {test_accuracy_extDWGM * 100:.2f}%")
print(f"Test accuracy with Gradient Descent: {test_accuracy_GD * 100:.2f}%")

