import numpy as np
from numpy.linalg import matrix_power

epsilon=0.000001
def test_convergence(v1,v2):
    diff=np.linalg.norm(v1/np.linalg.norm(v1)-v2/np.linalg.norm(v2))
    if(diff<=epsilon):
        return True
    else:
        return False

def principal_component(X,nbr):
    A=np.matmul(X.transpose(),X)
   
    shape_A=A.shape
    components=[]
    for component in range(nbr):
        v_0=np.random.rand(A.shape[0],1)#filling with random values
        if component!=0:
            for i in range(shape_A[1]):
                A[:,i]=A[:,i]-(np.dot(A[:,i],components[0])*components[0]).reshape(A[:,i].shape)
        v_i=v_0
        puissance=0
        converged=False
        while(converged==False):
            puissance+=1
            v_i_copy=v_i.copy()
            v_i=np.matmul(matrix_power(A,puissance),v_0)
            if test_convergence(v_i,v_i_copy)==True:
                converged=True
        print((v_i/np.linalg.norm(v_i)).shape)
        components.append((v_i/np.linalg.norm(v_i)).reshape((v_i.shape[0],)))
    return np.array(components)


A=np.array([[10.2,14,10],[4,55,20]])
#A=scale_matrix(np.matmul(A,A.transpose()))

v_0=np.random.rand(A.shape[0],1)#filling with random values
pca1=principal_component(A,2)
print(pca1)


#print(V)
#comparaison avec sklearn 

'''
from sklearn.decomposition import PCA
pca=PCA(2)
pca.fit(A)
print(pca.components_)
'''