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
    '''
    However, the above update rule may not satisfy that a singular
    value must be non-negative. This condition can easily be
    satisfied by multiplying either U or V with sgn(S) and taking
    the absolute values of S as a new S.

    '''
    U_t=U.transpose()
    
    interm=np.matmul(U_t,Y)
    S=np.matmul(interm,V)
    return S
    


def orthogolize(mat):#We need to check this

   mat_orth= sp.linalg.orth(mat)
   return mat_orth



import numpy as np
import random




def initialize(m,n):
    my_min=min(m,n)#il faut qu'il soit inférieur au min
    U=np.zeros((m,my_min))#init
    V=np.zeros((n,my_min))
    U=np.random.rand(m, my_min)#filling with random values
    V=np.random.rand(n, my_min)
    Y=np.random.rand(m,n)

    print("shape of U is",U.shape)
    print("shape of V is",V.shape)


    return U,Y,V

result=initialize(10,5)



def update_U_V(U,V,Y,eta,S):
    #U=U+eta*np.matmul((np.matmul(Y,V)+ np.matmul(np.matmul(np.matmul(U,V.transpose()),Y.transpose()),U),S))
    #V=V+eta*np.matmul((np.matmul(Y.transpose(),U)+ np.matmul(np.matmul(np.matmul(V,U.transpose()),Y),V),S))
    return U,V


def update_Y(Y,lbda,U,S,X,V):
    index_i=[i for i in range((X-Y).shape[0])]
    index_j=[j for j in range((X-Y).shape[1])]
    
    omega_set=set()
    for i,j in zip(index_i,index_j):
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
    lbda=0.01
    m=5
    n=3
    U,Y,V=initialize(2,3)
    eta=0.05
    K=2
    k=1
    tau=0.1
    S=s_computing(U,Y,V)    

    while iteration<100:
        update_U_V(U,V,Y,eta,S)#OK
        S=s_computing(U,Y,V)#OK
        S=turn_mat_positive(S)#OK but need to check the algorithm
        U,V=update_U_V(U,V,Y,eta,S)#we have to define Q_omega,lbda restricts  the amount of the distance between completed matrix Y and the originalmatrix , it's a penalization parameter
        Y=update_Y(Y,lbda,U,S,matrix,V)
        iteration=iteration+1
        #lambda should be determined based on the portion o the size of missing values
        #Update Y according to (9)
        lbdas=np.diag(Y)
        if n%n_r==0 and  check_sum_sv(K,k,lbdas,tau)==False:#lbdas= the sum of singular values estimated so fa
            #Append a set of random column vectors to U and V
            orthogolize(U)
            orthogolize(V)#We need to check this
            S=s_computing(U,Y,V)#OK
            S=turn_mat_positive(S)#OK but need to check the algorithm

        
    return U,Y
    

input_matrix=np.array([[1,2,3],[4,5,6]])
U,Y=SVD_algorithm(input_matrix,2,3,1)

print(U)

print('y=',Y)