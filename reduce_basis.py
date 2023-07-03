import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.io import loadmat
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import time
import collections.abc

def assembleA(mu,Aq):
    A=csc_matrix(mu[0]*Aq[0])
    for k in range(1,6):
        A+=mu[k]*Aq[k]
    return A

class finite_element():
    def __init__(self,type): #type = 'coarse','medium','fine'
        M = loadmat('FE_matrix.mat',simplify_cells=True)
        self.A_N_cal_q = M['FE_matrix'][type]['Ahq']
        self.N_cal = self.A_N_cal_q[0].shape[0]
        self.F_N_cal = M['FE_matrix'][type]['Fh']
        self.L_N_cal = self.F_N_cal

    def get_A_N_cal(self,mu):
        return assembleA(mu,self.A_N_cal_q)
    
    def get_u_N_cal(self,mu):
        A = self.get_A_N_cal(mu)
        return spsolve(A, self.F_N_cal)
    
    def get_T_root_N_cal(self,mu):
        return self.L_N_cal.T @ self.get_u_N_cal(mu)


class reduce_basis(finite_element):
    def __init__(self,type,sample,mu_bar,ortho=None,S_N=None): # nb_param = case
        super().__init__(type)

        self.A_prod = self.get_A_N_cal(mu_bar)

        # pour initialiser l'échantillon S_N
        if(sample==0): #greedy
            assert(ortho==None and isinstance(S_N, (collections.abc.Sequence, np.ndarray)))
            self.N = 1
            self.S_N = S_N 
            # self.S_N = np.array([mu_1])
            self.Z_complet = np.zeros((self.N_cal,1))
            self.Z_complet[:,0]=self.get_u_N_cal(self.S_N[0])
            self.Z_complet = self.__gramm_schmidt()
            assert np.max(np.abs(np.eye(self.N) - self.Z_complet.T @ self.A_prod @self.Z_complet)) < 1e-6

        elif(sample==4): # given sample
            assert(ortho!=None and isinstance(S_N, (collections.abc.Sequence, np.ndarray)))

            self.S_N = S_N 
            self.N = self.S_N.shape[0]

            self.Z_complet = self.__charge_Z()
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
        # Q = np.zeros((self.Z_complet.shape[0],self.Z_complet.shape[1]))
        # for i in range(self.Z_complet.shape[1]):
        i_last = self.Z_complet.shape[1]-1
        u = self.Z_complet[:,i_last]
        for j in range(i_last):
            u -= self.prod_scalaire(self.Z_complet[:,i_last],self.Z_complet[:,j])*self.Z_complet[:,j]
        self.Z_complet[:,i_last] = u/self.norme(u)

    def one_step_greedy(self,method,method_name,mu_train,normalize=True):
        if method_name=="offline_online":
            Q_1,Q_2,Q_3 = method.offline_quantity()
            tab_Delta_N_en = [method.compute_Delta_N_en(mu,Q_1,Q_2,Q_3) for mu in mu_train]
        elif method_name=="direct":
            tab_Delta_N_en = [method.compute_Delta_N_en(mu) for mu in mu_train]

        tab_A_N_cal_mu = [self.get_A_N_cal(mu) for mu in mu_train]
        tab_Zu_N = [self.Z_complet@self.get_u_N(mu) for mu in mu_train]
        w_mu = [np.sqrt(tab_Zu_N[i].T @ tab_A_N_cal_mu[i] @ tab_Zu_N[i]) for i in range(len(mu_train))]

        if normalize==True:
            den = w_mu
        else:
            den = np.ones(len(mu_train))
        
        val = [tab_Delta_N_en[i]/den[i] for i in range(len(mu_train))]
        index = np.nanargmax(val) # pour ne pas prendre en compte les nan
        mu_N = mu_train[index]
        Delta_N_max = val[index]

        self.N += 1
        self.S_N = np.append(self.S_N,[mu_N],axis=0)
        self.Z_complet = np.concatenate((self.Z_complet,self.get_u_N_cal(mu_N)[:,np.newaxis]),axis=1)
        # self.Z_complet = self.__gramm_schmidt()
        self.__one_step_gramm_schmidt()
        assert np.max(np.abs(np.eye(self.N) - self.Z_complet.T @ self.A_prod @self.Z_complet)) < 1e-6
        self.Z = self.Z_complet

        self.A_N_q = self.__charge_A_N_q()
        self.F_N = self.Z_complet.T @ self.F_N_cal
        self.L_N = self.F_N

        return Delta_N_max,mu_N
        
    
    def __charge_A_N_q(self):
        A_N_q = []
        for i in range(6):
            A_N_q.append(self.Z.T @ self.A_N_cal_q[i] @ self.Z)
        return np.array(A_N_q)
    
    def get_A_N(self,mu):
        return assembleA(mu,self.A_N_q)

    def get_u_N(self,mu): # u_N(mu)
        A = self.get_A_N(mu)
        return np.linalg.solve(A, self.F_N)
    
    def get_T_root_N(self,mu):
        return self.L_N.T @ self.get_u_N(mu)

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
        self.L_N = self.F_N