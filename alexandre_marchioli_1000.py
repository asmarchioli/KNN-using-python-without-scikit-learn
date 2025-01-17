import numpy as np
import pandas as pd
from collections import defaultdict

def knn(k, dataset, dataexemplos):
    '''O código não faz uma verificação explícita de empate. Em vez disso, usa um
    sistema de votação ponderada, onde a contribuição de cada vizinho para a
    decisão é inversamente proporcional à sua distância do ponto de teste. A
    classe que tem a maior soma ponderada de votos é escolhida como a
    classificação final para o ponto de teste. Assim, a ponderação por distância
    ajuda a evitar a necessidade de um desempate explícito, pois vizinhos mais
    próximos têm mais influência na decisão final.'''
    INDICES = []
    dataexemplos_list = dataexemplos.values.tolist()
    database_list = dataset.drop(['Label'], axis=1).values.tolist()
    database_lb_list = dataset['Label'].values.tolist()

    for e_exemplos in dataexemplos_list:
        DISTANCIAS = []

        for e_treino, label in zip(database_list, database_lb_list): #A função zip junta duas listas em pares (precisam ter o mesmo tamanho)
            d_soma = np.sqrt(np.sum((np.array(e_treino) - np.array(e_exemplos))**2)) #Faz todas as contas direto nos arrays, o que agiliza o processamento do código
            DISTANCIAS.append((d_soma, label))

        #Ordena as listas de distancias e labels pelas distancias
        DISTANCIAS.sort(key=lambda x: x[0])

        #Obtem os k vizinhos mais próx
        k_vizinhos = DISTANCIAS[:k]

        contagem = defaultdict(float) #defaultdict -> subclasse do dict que cria um tipo float "0.0" para cada valor que não existir e for chamado
        for distancia, label in k_vizinhos:
            contagem[label] += 1 / (distancia + 1e-5)  # Adiciona um pequeno valor para evitar divisão por zero + distâncias menores tem mais peso

        #Encontrar a classe com a maior contagem
        mais_comum = max(contagem, key=contagem.get)
        INDICES.append(mais_comum)

    return INDICES
