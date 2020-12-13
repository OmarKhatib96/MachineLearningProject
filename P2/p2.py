import pandas as pd
import seaborn as sn

import matplotlib.pyplot as plt



def scale_matrix(A):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(A)
    A=scaler.transform(A)
    return A



pd.set_option('display.max_colwidth', -1)#tro 



d0 = pd.read_csv('myocardy_complications.csv')
from sklearn.impute import SimpleImputer
import numpy as np





imp=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imp.fit(d0)
d0_imputed=imp.transform(d0)


y=d0_imputed[:,123]
d0_imputed =np.delete(d0_imputed,-1,axis=1)
print(d0_imputed.shape)
#with pd.option_context('display.max_columns', None):  # more options can be specified also
    #print(d0.head(1))




'''

#cat=d0.select_dtypes(include='object').columns

print(d0['LET_IS'].dtype)

import seaborn as sns
#sns.kdeplot(d0['AGE'],shade=True)
sns.scatterplot(x=d0['AGE'],y=d0['GB'],data=d0)

plt.show()


print(d0.head(1).transpose()) # print first five rows of d0.
#print(d0.tail(10))
# save the labels into a variable l.
print('shape of data set=',d0.shape)
print('size of data set=',d0.size)


d0.info()
d0.describe()
#at=d0.select_dtypes(include='object').columns
#print(cat)

print(list(d0))

print(d0.isnull().sum())#count missing values per columns
#sn.countplot(d0['SEX'])
#sn.countplot(d0['AGE'])
sn.barplot(d0['AGE'],d0['JELUD_TAH'])
plt.show()
#imputing missing values
'''

#To count the number 
nbr_null_elements=np.count_nonzero(np.isnan(d0_imputed))
print(d0_imputed[1])
print(nbr_null_elements)








# initializing the pca
from sklearn import decomposition

pca = decomposition.PCA()

# configuring the parameteres
# the number of components = 2
pca.n_components = 10
print(d0_imputed.shape)
pca_data = pca.fit_transform(d0_imputed)
# pca_reduced will contain the 2-d projects of simple data
print("shape of pca_reduced.shape = ", pca_data.shape)

labels=["unknown","cardiogenic shock","pulmonary edema","myocardial rupture","progress of congestive heart failure","thromboembolism","asystole","ventricular fibrillation"]
# attaching the label for each 2-d data point
dic_legend={0.0:"unknown",1.0:"cardiogenic shock",2.0:"pulmonary edema",3.0:"myocardial rupture",4.0:"progress of congestive heart failure",5.0:"thromboembolism",6.0:"asystole",7.0:"ventricular fibrillation"}
pca_data_stacked = np.vstack((pca_data[:,0:2].T, y)).T


import matplotlib 
from matplotlib import legend

import numpy as np
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D



import seaborn as sn
# creating a new data fram which help us in ploting the result data
pca_df = pd.DataFrame(data=pca_data_stacked, columns=("1st_principal", "2nd_principal","label"))

g=sn.FacetGrid(pca_df,hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend(title="Cause of mortality")
g.fig.suptitle("PCA on causes of death involving myocardial complication")

plt.show()



from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=7, random_state=0).fit(pca_data)


predicted_labels=kmeans.labels_
print(predicted_labels)
pca_data_stacked_kmeans = np.vstack((pca_data[:,0:2].T, predicted_labels)).T

pca_df = pd.DataFrame(data=pca_data_stacked, columns=("1st_principal", "2nd_principal","predicted_labels"))
g=sn.FacetGrid(pca_df,hue="predicted_labels", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend(title="Cause of mortality")
plt.legend(loc='lower right')

plt.show()






#legend(y,[labels[y[i]] for i in range(y[i])])



#Estimation of the intrinsic dimensionality

import skdim
#d0=scale_matrix(d0)
#estimate global intrinsic dimension
danco = skdim.id.DANCo().fit(d0_imputed)
print(danco.dimension_)

import seaborn as sns
sns.countplot(y)
plt.show()
