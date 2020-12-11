import numpy as np
epsilon=0.0001

def iteration(A,index,first_component):
    v_iteration=np.random.rand(A.shape[0],1)#filling with random values
    shape_A=A.shape



    if index!=0:
        for i in range(shape_A[1]):
            A[:,i]=A[:,i]-first_component*np.dot(A[:,i],first_component)
    

    #for it in range(10000):
    converged=False
    while(converged==False):
        v_iteration_copy=v_iteration.copy()
        v_iteration=v_iteration/np.linalg.norm(v_iteration)

        u_i=[]
        for i in range(shape_A[1]):
            #u_i.append(np.dot(A[:,i],v_iteration)*A[:,i])
            u_i.append(np.dot(A[:,i],v_iteration))


        sum_u_i_x_i=0
        sum_ui=0

        for i in range(shape_A[1]) :
            sum_u_i_x_i+=A[:,i]*u_i[i]
            sum_ui+=u_i[i]**2
        v_iteration=sum_u_i_x_i/sum_ui
        #print(np.linalg.norm(v_iteration_copy-v_iteration))
        if(np.linalg.norm(v_iteration_copy-v_iteration)<=epsilon):
            converged=True

    return v_iteration

def iterative_svd(A):
    v1=np.random.rand(A.shape[0],1)#filling with random values
    first_component=v1
    v_i=[]
    for index in range(A.shape[1]):
        element= iteration(A,index,first_component)
        if index==0:
            first_component=element
        
        #print("element=",element)
        v_i.append(element)

    V=np.array(v_i)
    return V
        

def scale_matrix(A):

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(A)
    A=scaler.transform(A)
    return A



A=np.array([[1,0,105],[4,5,20]])
#A=scale_matrix(np.matmul(A,A.transpose()))
V=iterative_svd(np.matmul(A.transpose(),A))


print(V)

#comparaison avec sklearn     
from sklearn.decomposition import PCA
pca=PCA(2)
pca.fit(A)
print(pca.components_)
