'''
Algorithm 1 The iterative algorithm for computing SVD.
Input an input matrix Xm×n, a set of observed indices Ω,
a rank-update interval nr
Initialize Um×r, Vn×r and Ym×n to some small random
values where r < min(m, n) is a small integer.
Orthogonalize U and V according to (5)
Compute S according to (8)
Set n = 0
repeat
Update U and V according to (6) and (7)
Update S according to (8)
Make all sii positive according to Section II-B
Update Y according to (9).
if mod (n, nr) = 0 and (10) is false then
Append a set of random column vectors to U and V
Orthogonalize U and V
Update S
Make all sii positive
end if
until Stopping criterion is reached
'''
import scipy as sp
import numpy as np

eta=2


def turn_mat_positive(M):
    M = np.maximum(M, -M) 

    return M#Have to check it later idk


def s_computing(U,Y,V):#OK
   
    U_t=U.transpose()
    interm=np.matmul(U_t,Y)
    S=np.matmul(interm,V)
    return S
    


def orthogolize(mat):#We need to check this
   from scipy import linalg
   mat_orth= linalg.orth(mat)
   return mat_orth



import numpy as np
import random




def initialize(m,n):
    my_min=min(m,n)#il faut qu'il soit inférieur au min
    U=np.random.rand(m, my_min)#filling with random values
    V=np.random.rand(n, my_min)
    Y=np.zeros((m,n))#init 
    #filling the diagonal
    vec=np.random.rand(my_min)
    for i in range(my_min):
        Y[i,i]=vec[i]
    print("shape of U is",U.shape)
    print("shape of V is",V.shape)
    return U,Y,V




def update_U_V(U,V,Y,eta,S):
    Y_V=(np.matmul(Y,V))
    U_V_T=np.matmul(U,V.transpose())
    Y_T_U=np.matmul(Y.transpose(),U)
    V_U_T=np.matmul(V,U.transpose())

    
    U=U+eta*(Y_V+np.matmul(np.matmul(U_V_T,Y_T_U),S))
    V=V+eta*(Y_T_U+np.matmul(np.matmul(V_U_T,Y_V),S))
    return U,V


def update_Y(Y,lbda,U,S,X,V):
    #full set observed
    omega_set=set()
    for i in range((X-Y).shape[0]):
        for j in range((X-Y).shape[1]):
            omega_set.add((i,j))

    U_S=np.matmul(U,S)
    Y=Y+eta*(np.matmul(U_S,V.transpose())-Y+lbda*omega_matrix(X-Y,omega_set))
    return Y


def omega_matrix(Z,omega_set):
    omega_mat=np.zeros((Z.shape[0],Z.shape[1]))
    index_i=[i for i in range(Z.shape[0])]
    index_j=[j for j in range(Z.shape[1])]

    for i,j in zip(index_i,index_j):
        if (i,j) in omega_set:
            omega_mat[i,j]=Z[i,j]
        else:
            omega_mat[i,j]=0
    
    return omega_mat



def check_sum_sv(K,k,lbdas,tau):
    sum_lbdas=np.sum(lbdas)
    if (sum_lbdas/(sum_lbdas+(K-k)*lbdas[k])>tau):
        return True
    else:
        return False

def SVD_algorithm(matrix,m,n,n_r):
    n=0#we set n =0 as it is said in the pseudoalgorithm
    iteration=0
    lbda=1
    m=5
    n=3
    U,Y,V=initialize(m,n)
    eta=0.05
    K=2
    k=1
    tau=0.5
    print('Y=',Y)
    U=orthogolize(U)
    V=orthogolize(V)
    S=s_computing(U,Y,V)    
    N=0
    while iteration<2:
        U,V=update_U_V(U,V,Y,eta,S)#OK
        U=orthogolize(U)
        V=orthogolize(V)#We need to check this
        S=s_computing(U,Y,V)#OK
        S=turn_mat_positive(S)#OK but need to check the algorithm
        print("S=",S)
        Y=update_Y(Y,lbda,U,S,matrix,V)
        iteration=iteration+1
        #lambda should be determined based on the portion o the size of missing values
        #Update Y according to (9)
        lbdas=np.diag(Y)
        if N%n_r==0 and  check_sum_sv(K,k,lbdas,tau)==False:#lbdas= the sum of singular values estimated so fa
            #Append a set of random column vectors to U and V
            print("hh")
            U=orthogolize(U)
            V=orthogolize(V)#We need to check this
            S=s_computing(U,Y,V)#OK
            S=turn_mat_positive(S)#OK but need to check the algorithm

    return U,Y,S,V

m=5
n=3
X=np.random.rand(m, n)

U,Y,S,V=SVD_algorithm(X,2,3,1)

print("U=",U)
print("Y=",Y)
print("V=",Y)
print('S=',S)

