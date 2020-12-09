import numpy as np
import scipy as scipy


def svd_decomposition(A):
    A_T_A=np.matmul(A.transpose(),A)
    eigen_values,eigen_vectors=np.linalg.eig(A_T_A)
    idx=eigen_values.argsort()[::-1]
    eigen_values=eigen_values[idx]
    eigen_vectors=eigen_vectors[:,idx]
    nbr_vectors=len(eigen_values)
    V=[]
    for vector in range(0,nbr_vectors):
        V.append(eigen_vectors[vector])
    V=np.array(V).transpose()
    eigen_values=np.sqrt(np.abs(eigen_values))
    S=np.diag(eigen_values)
    S_inv=np.linalg.inv(S)
    U=np.matmul(A,np.matmul(V,np.linalg.inv(S)))
    return U,S,V






A=np.array([[1,2,10],[4,5,20],[7,8,9]])

U,S,V=svd_decomposition(A)
print("Produit matrice=",np.matmul(U,np.matmul(S,V.transpose())))