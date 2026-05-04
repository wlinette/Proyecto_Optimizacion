import method_functions as meth_f
import loss_functions as loss_f
import numpy as np
import matplotlib.pyplot as plt


epsilon = 1e-8
n = 1000 
x0 = np.array([2.0]*n, dtype=np.float64)
xk_list, gk_norm_list, fk_list, k_list = meth_f.extended_DWGM(loss_f.f_func ,loss_f.grad_func, loss_f.hessian_func , loss_f.ek_func , x0, t = 1 ,max_iter = 10000, epsilon = epsilon,delta = 0.9 ,gamma = 1e-4, m = 100)


path_to_save = "results/"
fig_name = "grafica_exp1.png"
fig = plt.figure(figsize=(10, 6))
plt.plot(k_list, gk_norm_list)
plt.yscale('log')
plt.xlabel('Iteración k')
plt.ylabel('Norma del gradiente ||g_k||')
plt.title('Norma del gradiente a lo largo de las iteraciones')
plt.grid()

plt.savefig(path_to_save + fig_name, dpi=300, bbox_inches='tight')