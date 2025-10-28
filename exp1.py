import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
# Load dataset (Dataset want to be downloaded) 
df = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")

print(df.shape) 
df.head(150) 
# Univariate analysis for sepal width 
df_Setosa = df.loc[df['species'] == 'setosa']
df_Virginica = df.loc[df['species'] == 'virginica']
df_Versicolor = df.loc[df['species'] == 'versicolor']


plt.scatter(df_Setosa['sepal_width'], np.zeros_like(df_Setosa['sepal_width']), 
label='Setosa') 
plt.scatter(df_Virginica['sepal_width'], np.zeros_like(df_Virginica['sepal_width']), 
label='Virginica') 
plt.scatter(df_Versicolor['sepal_width'], np.zeros_like(df_Versicolor['sepal_width']), 
label='Versicolor') 
plt.xlabel('sepal.width') 
plt.legend() 
plt.show() 
# Univariate analysis for sepal length 
plt.scatter(df_Setosa['sepal_length'], np.zeros_like(df_Setosa['sepal_length']), 
label='Setosa') 
plt.scatter(df_Virginica['sepal_length'], np.zeros_like(df_Virginica['sepal_length']), 
label='Virginica') 
plt.scatter(df_Versicolor['sepal_length'], np.zeros_like(df_Versicolor['sepal_length']), 
label='Versicolor') 
plt.xlabel('sepal.length') 
plt.legend() 
plt.show() 
# Univariate analysis for petal width 
plt.scatter(df_Setosa['petal_width'], np.zeros_like(df_Setosa['petal_width']), 
label='Setosa') 
plt.scatter(df_Virginica['petal_width'], np.zeros_like(df_Virginica['petal_width']), 
label='Virginica') 
plt.scatter(df_Versicolor['petal_width'], np.zeros_like(df_Versicolor['petal_width']), 
label='Versicolor') 
plt.xlabel('petal.width') 
plt.legend() 
plt.show() 
# Univariate analysis for petal length 
plt.scatter(df_Setosa['petal_length'], np.zeros_like(df_Setosa['petal_length']), 
label='Setosa') 
plt.scatter(df_Virginica['petal_length'], np.zeros_like(df_Virginica['petal_length']), 
label='Virginica') 
plt.scatter(df_Versicolor['petal_length'], np.zeros_like(df_Versicolor['petal_length']), 
label='Versicolor') 
plt.xlabel('petal.length') 
plt.legend() 
plt.show() 
# Bivariate analysis using FacetGrid 
sns.FacetGrid(df, hue='species', height=5).map(plt.scatter, "sepal_width", 
"petal_width").add_legend() 
plt.show() 
sns.FacetGrid(df, hue='species', height=5).map(plt.scatter, "sepal_length", 
"petal_length").add_legend() 
plt.show() 
# Multivariate analysis (pairplot) 
sns.pairplot(df, hue="species", height=2) 
plt.show()
