import pandas as pd
from alexandre_marchioli_1000 import knn

#Mudar para a pasta local dos CSVs
dataexemplos = pd.read_csv('Aula 18 - dataexemplos.csv')
dataset = pd.read_csv('Aula 18 - dataset.csv')
dataset = dataset.fillna(0)

#Valor do número de vizinhos próximos
k = 5

print(knn(k, dataset, dataexemplos))