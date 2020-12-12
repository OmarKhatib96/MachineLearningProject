import numpy as np
epsilon=0.0001




# MNIST dataset downloaded from Kaggle : 
#https://www.kaggle.com/c/digit-recognizer/data

# Functions to read and show images.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


   
d0 = pd.read_csv('train_data.csv')
print(d0.head(5)) # print first five rows of d0.
# save the labels into a variable l.
l = d0['label']
# Drop the label feature and store the pixel data in d.
d = d0.drop("label",axis=1)


print(d.shape)
print(l.shape)



# display or plot a number.
plt.figure(figsize=(7,7))
idx = 500

grid_data = d.iloc[idx].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array
plt.imshow(grid_data, interpolation = "none", cmap = "gray")
plt.show()

print(l[idx])



# Pick first 15K data-points to work on for time-effeciency.
#Excercise: Perform the same analysis on all of 42K data-points.

labels = l.head(15000)
data = d.head(15000)

print("the shape of sample data = ", data.shape)






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

def iterative_svd(X,nbr_components):
    A=np.matmul(X.transpose(),X)
    if nbr_components<=min(A.shape[0],A.shape[1]):
        v1=np.random.rand(A.shape[0],1)#filling with random values
        first_component=v1
        v_i=[]
        for index in range(nbr_components):
            element= iteration(A,index,first_component)
            if index==0:
                first_component=element
            else:
                first_component=v_i[index-1]
            
            #print("element=",element)
            v_i.append(element)

        V=np.array(v_i)
        return V
    else:
        print("the number of components should be equal or less  to min(dim_features,dim_samples)")
        return []
        

def scale_matrix(A):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(A)
    A=scaler.transform(A)
    return A




# projecting the original data sample on the plane 
#formed by two principal eigen vectors by vector-vector multiplication.

import matplotlib.pyplot as plt

def project_data_on_components(principal_components):
    new_coordinates = np.matmul(principal_components, sample_data.T)
    return new_coordinates


def data_visualize(coordinates):

    import pandas as pd
    import seaborn as sn

    coordinates = np.vstack((coordinates, labels)).T#Add labels to the vector 
    dataframe = pd.DataFrame(data=coordinates, columns=("1st_principal", "2nd_principal", "label"))
    print(dataframe.head())
    sn.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
    plt.show()







def data_scaling(data):
    from sklearn.preprocessing import StandardScaler
    standardized_data = StandardScaler().fit_transform(data)
    print("standardized data shape=",standardized_data.shape)
    return standardized_data



#I-Data scaling
standardized_data=data_scaling(data)
sample_data = standardized_data
'''
#II-Principal components computation
components=iterative_svd(sample_data,4)
print(components.shape)
#III-projection of original data on the PC
projection_coordinates=project_data_on_components(components[0:2])
#IV-Data Visualization of projection
data_visualize(projection_coordinates)

'''
#comparaison avec sklearn 

# initializing the pca
from sklearn import decomposition
pca = decomposition.PCA()

# configuring the parameteres
# the number of components = 2
pca.n_components = 4
print(sample_data.shape)
print(standardized_data)
pca_data = pca.fit_transform(standardized_data)
# pca_reduced will contain the 2-d projects of simple data
print("shape of pca_reduced.shape = ", pca_data.shape)


# attaching the label for each 2-d data point 
pca_data = np.vstack((pca_data[:,1:3].T, labels)).T
import seaborn as sn
# creating a new data fram which help us in ploting the result data
pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "label"))
sn.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()




#II-Principal components computation
components=iterative_svd(sample_data,2)
print(components.shape)
#III-projection of original data on the PC
projection_coordinates=project_data_on_components(components)
#IV-Data Visualization of projection
data_visualize(projection_coordinates)

