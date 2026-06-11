import pandas as pd

winedatac=pd.read_csv('D:/4050/DataSets/wine.csv')

print(winedatac)
print(winedatac.head())
print("shape\n",winedatac.shape)
print("columns\n",winedatac.columns)
print("dtypes\n",winedatac.dtypes)
print("ndim\n",winedatac.ndim)
print("size\n",winedatac.size)


winedatae=pd.read_excel('D:/4050/DataSets/wine.xls')

print('\n')


