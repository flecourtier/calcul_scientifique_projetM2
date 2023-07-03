import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import spsolve
from scipy.io import loadmat
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import time
import collections.abc

def compute_theta_q_mu(q,mu):
    assert(1<=q and q<=6)
    return mu[q-1]

def assembleA(mu,Aq):
    A=csc_matrix(mu[0]*Aq[0])
    for k in range(1,6):
        A+=mu[k]*Aq[k]
    return A

# dépendant du temps
class finite_element_time():
    def __init__(self,type): #type = 'coarse','medium','fine'
        self.M_N_cal = loadmat('FE_matrix_mass.mat',simplify_cells=True)['FE_matrix_mass'][type]['Mh']
        self.L = self.M_N_cal @ np.ones(self.M_N_cal.shape[0])

        load_file = loadmat('FE_matrix.mat',simplify_cells=True)

        self.A_N_cal_q = load_file['FE_matrix'][type]['Ahq']
        self.N_cal = self.A_N_cal_q[0].shape[0]

        self.F_N_cal = load_file['FE_matrix'][type]['Fh']
        self.L_N_cal = self.F_N_cal

    def get_u_N_cal_k(self, U0, g, dt, K, mu):
        A_N_cal = self.get_A_N_cal(mu)
        B = 1/dt * self.M_N_cal + A_N_cal
        SuperLU_B = splu(B)
        # to check the LU decomposition
        Pr = csc_matrix((np.ones(self.N_cal), (SuperLU_B.perm_r, np.arange(self.N_cal))))
        Pc = csc_matrix((np.ones(self.N_cal), (np.arange(self.N_cal), SuperLU_B.perm_c)))
        assert(np.max(np.abs(B - (Pr.T @ (SuperLU_B.L @ SuperLU_B.U) @ Pc.T).A)) < 1e-6)
        tk=0.
        tab_Uk = [U0]
        for k in range(1,K+1):
            g_tk = g(tk,k)
            rhs = self.F_N_cal * g_tk + 1/dt * self.M_N_cal @ tab_Uk[-1]
            Uk = SuperLU_B.solve(rhs)
            tab_Uk.append(Uk)
            tk += dt
        return np.array(tab_Uk)

    def get_T_root_N_cal_k(self, U0, g, dt, K, mu):
        tab_U_k_N_cal = self.get_u_N_cal_k(U0, g, dt, K, mu)
        s_k_N_cal = [self.L_N_cal.T @ u_k_N_cal for u_k_N_cal in tab_U_k_N_cal]
        return np.array(s_k_N_cal), tab_U_k_N_cal

    def get_A_N_cal(self,mu):
        return assembleA(mu,self.A_N_cal_q)
    
    def get_u_N_cal(self,mu):
        A = self.get_A_N_cal(mu)
        return spsolve(A, self.F_N_cal)
    
    def get_T_root_N_cal(self,mu):
        return self.L_N_cal.T @ self.get_u_N_cal(mu)


class reduce_basis_time(finite_element_time):
    def __init__(self,type,sample,mu_bar,ortho=None,S_N=None,X_N=None): # nb_param = case
        super().__init__(type)

        self.A_prod = self.get_A_N_cal(mu_bar)

        # pour initialiser l'échantillon S_N
        if(sample==0): #greedy
            assert(ortho==None and isinstance(S_N, (collections.abc.Sequence, np.ndarray)))
            self.N = 0
            self.S_N = []
            self.Z_complet = np.zeros((self.N_cal,1))

        elif(sample==4): # given sample S_N
            assert(ortho!=None and isinstance(S_N, (collections.abc.Sequence, np.ndarray)))

            self.S_N = S_N 
            self.N = self.S_N.shape[0]

            self.Z_complet = self.__charge_Z()
            if ortho==True:
                self.Z_complet = self.__gramm_schmidt()

        elif(sample==5): # given basis X_N (no sample S_N)
            assert(ortho!=None and isinstance(X_N, (collections.abc.Sequence, np.ndarray)))

            self.Z_complet = X_N
            self.N = self.Z_complet.shape[1]
            if ortho==True:
                self.Z_complet = self.__gramm_schmidt()

        else:
            assert(ortho!=None and S_N==None)
            self.S_N = self.__charge_S_N(sample)
            self.Z_complet = self.__charge_Z()
            
            if ortho==True:
                self.Z_complet = self.__gramm_schmidt()
                #check orthonormality
                assert np.max(np.abs(np.eye(self.N) - self.Z_complet.T @ self.A_prod @self.Z_complet)) < 1e-6
        self.Z = self.Z_complet

        self.A_N_q = self.__charge_A_N_q()
        self.F_N = self.Z.T @ self.F_N_cal
        self.L_N = self.F_N
        self.M_N = self.Z.T @ self.M_N_cal @ self.Z
    
    def __charge_S_N(self,s):
        sample = loadmat('RB_sample.mat',simplify_cells=True)['RB_sample']['sample'+str(s)]
        self.N = sample.shape[0]
        if s==3:
            self.N = sample[0].shape[0]
        
        S_N=[]
        for i in range(self.N):
            if s==1:
                k_i = sample[i]
                mu_i = [k_i,k_i, k_i, k_i, 1, 0.1]
            elif s==2:
                Bi = sample[i]
                mu_i = [0.4, 0.6, 0.8, 1.2, 1, Bi]
            elif s==3:
                k_i = sample[0][i]
                Bi = sample[1][i]
                mu_i = [k_i,k_i, k_i, k_i, 1, Bi]
            S_N.append(mu_i)
        return np.array(S_N)

    def get_u_N_k(self, U0, g, dt, K, mu):
        A_N = self.get_A_N(mu)
        B = 1/dt * self.M_N + A_N
        lu, piv = lu_factor(B)
        # to check the LU decomposition

        # L, U = np.tril(lu, k=-1) + np.eye(self.Z.shape[1]), np.triu(lu)
        # print(np.max(np.abs(B[piv] - L @ U)))
        # assert(np.max(np.abs(B[piv] - L @ U)) < 1e-6)
        tk=0.
        tab_Uk = [U0]
        for k in range(1,K+1):
            g_tk = g(tk,k)
            rhs = self.F_N * g_tk + 1/dt * self.M_N @ tab_Uk[-1]
            Uk = lu_solve((lu, piv), rhs)
            tab_Uk.append(Uk)
            tk += dt
        return np.array(tab_Uk)

    def get_T_root_N_k(self, U0, g, dt, K, mu):
        tab_U_k_N = self.get_u_N_k(U0, g, dt, K, mu)
        s_k_N = [self.L_N.T @ u_k_N for u_k_N in tab_U_k_N]
        return np.array(s_k_N),tab_U_k_N

    def __charge_Z(self):
        Z = np.zeros((self.N_cal,self.N))
        for i in range(self.N):
            Z[:,i]=self.get_u_N_cal(self.S_N[i])
        return Z

    def __gramm_schmidt(self):
        Q = np.zeros((self.Z_complet.shape[0],self.Z_complet.shape[1]))
        for i in range(self.Z_complet.shape[1]):
            u = self.Z_complet[:,i]
            for j in range(i):
                u -= self.prod_scalaire(self.Z_complet[:,i],Q[:,j])*Q[:,j]
            Q[:,i] = u/self.norme(u)
        return Q

    def __one_step_gramm_schmidt(self): # pour orthonormaliser la dernière colonne
        i_last = self.Z_complet.shape[1]-1
        u = self.Z_complet[:,i_last]
        for j in range(i_last):
            u -= self.prod_scalaire(self.Z_complet[:,i_last],self.Z_complet[:,j])*self.Z_complet[:,j]
        self.Z_complet[:,i_last] = u/self.norme(u)

    def __POD_X(self,e_k_N_proj,tab_U_k_N_cal_mu_N):
        K = len(e_k_N_proj)
        corr_matrix = np.zeros((K,K))
        for i in range(K):
            for j in range(K):
                corr_matrix[i][j] = 1/K * self.prod_scalaire(e_k_N_proj[i],e_k_N_proj[j])
        lambda_POD_k, psi_POD_k = np.linalg.eig(corr_matrix)
        index = np.argmax(lambda_POD_k)
        psi_POD_max = psi_POD_k[index]
        psi_POD1 = psi_POD_max[0] * tab_U_k_N_cal_mu_N[0]
        for k in range(1,K):
            psi_POD1 += psi_POD_max[k] * tab_U_k_N_cal_mu_N[k]
        return psi_POD1

    def one_step_greedy(self,method,mu_train,g,dt,K,mu_N): # method de type offline_online_method_time
        # U0, g, dt, tab_U_k_N, k, mu, quantity
        U0 = np.zeros(self.N_cal)
        tab_U_k_N_cal_mu_N = self.get_u_N_cal_k(U0, g, dt, K, mu_N)
        if(self.N==0):
            e_k_N_proj = tab_U_k_N_cal_mu_N
        else:
            e_k_N_proj = [U_k_N_cal_mu_N - self.Z_complet @ (self.Z_complet.T @ self.A_prod @ U_k_N_cal_mu_N) for U_k_N_cal_mu_N in tab_U_k_N_cal_mu_N]

        psi_POD1 = self.__POD_X(e_k_N_proj,tab_U_k_N_cal_mu_N)

        if(self.N==0):
            self.S_N = np.array([mu_N])
            self.Z_complet[:,0] = psi_POD1
        else:
            self.S_N = np.append(self.S_N,[mu_N],axis=0)
            self.Z_complet = np.concatenate((self.Z_complet,psi_POD1[:,np.newaxis]),axis=1)
        self.N += 1
        self.__one_step_gramm_schmidt()
        assert np.max(np.abs(np.eye(self.N) - self.Z_complet.T @ self.A_prod @self.Z_complet)) < 1e-6
        self.Z = self.Z_complet

        quantity = method.offline_quantity()
        val = []
        U0_N = np.zeros(self.N + (int)(self.N==0))
        for mu in mu_train:
            A_N_cal_mu = self.get_A_N_cal(mu)
            tab_U_k_N = self.get_u_N_k(U0_N, g, dt, K, mu)
            U_N_K = tab_U_k_N[-1]
            Delta_N_K = method.compute_Delta_N_en(U0_N, g, dt, tab_U_k_N, K, mu, quantity)
            val.append(Delta_N_K/np.sqrt(U_N_K.T @ A_N_cal_mu @ U_N_K))
        index = np.nanargmax(val)
        mu_N = mu_train[index]
        Delta_N_max = val[index]
            
        self.A_N_q = self.__charge_A_N_q()
        self.F_N = self.Z_complet.T @ self.F_N_cal

        return Delta_N_max,mu_N   
    
    def __charge_A_N_q(self):
        A_N_q = []
        for i in range(6):
            A_N_q.append(self.Z.T @ self.A_N_cal_q[i] @ self.Z)
        return np.array(A_N_q)
    
    def get_A_N(self,mu):
        return assembleA(mu,self.A_N_q)

    def prod_scalaire(self,u,v):
        return u.T @ self.A_prod @ v

    def norme(self,u):
        return np.sqrt(self.prod_scalaire(u,u))
    
    def change_N(self,N):
        if N!=0 and N<self.N:
            self.Z = self.Z_complet[:,:N]
        else:
            self.Z=self.Z_complet
        self.A_N_q = self.__charge_A_N_q()
        self.F_N = self.Z.T @ self.F_N_cal
        self.M_N = self.Z.T @ self.M_N_cal @ self.Z
        self.L_N = self.F_N
    
    def get_u_N(self,mu): # u_N(mu)
        A = self.get_A_N(mu)
        return np.linalg.solve(A, self.F_N)
    
    def get_T_root_N(self,mu):
        return self.L_N.T @ self.get_u_N(mu)