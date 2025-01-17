# kNN e seu funcionamento
O kNN, ou do inglês, K-Nearest Neighbors (K-nésimos vizinhos mais próximos), é um algoritmo (conhecido por ser um algoritmo preguiçoso - lazy algorithm) de instância utilizado para realizar uma previsão de classe/rótulo de elementos com certos parâmetros uma amostra de dados. Resumidamente ele calcula a distância - que pode ser euclidiana, de Chebyshev, de Manhattan, etc - de cada ponto n-dimensional formado pelos n parâmetros (n indica o número de parâmetros descontando a classe), e é adicionado o rótulo da instância mais próxima ao elemento sendo testado.

Por exemplo, a seguinte tabela possui notas de alunos e sua classe pertencente:

| Nota 1        | Nota 2       | Rótulo        |
| :-----------: | :----------: | :-----------: |
|       3       |       7      |        B      |
|       8       |       7      |        A      |
|       1       |       5      |        B      |
|       9       |       6      |        A      |
(sendo k = 1)

Se adicionarmos um novo aluno, com notas 9 e 8, sem rótulo, podemos o rotular com os exemplos anteriores medindo a distância de cada elemento da tabela. Como há 2 parâmetros (n = 2), então o campo será bidimensional, isso é, um Plano Cartesiano, o que torna mais fácil de visualizar que o valor a ser atribuido a classe deste novo aluno será "A".

Quando o valor de k é maior que 1, o rótulo de elementos com as distâncias mais próximas em maior número são testados. Em caso de empate dos rótulos, um critério de desempate precisa ser adicionado, como a distância mais próxima, o rótulo mais usado, etc.

Além da distância, do valor de k e do critério de desempate, outros padrões pode ser alterados, como os pesos dos valores, parâmetros métricos, entre outros.

# Implementação própria do kNN
Para minha implementação própria, os valores de k e os conjuntos csv são dados no começo do código. É necessário, antes de tudo, alterar os locais dos CSVs para onde estiverem e o valor de k, caso queira. Por padrão, o k começa como 5.
    
Procurei utilizar a biblioteca Numpy além do Pandas. Ela se tornou muito útil principalmente para melhor desempenho no cálculo de distâncias, que são euclidianas, por já possuir um método para isso. Utilizei o seguinte para o cálculo das distâncias:

    d_soma = np.sqrt(np.sum((np.array(e_treino) - np.array(e_exemplos))**2))
    DISTANCIAS.append((d_soma, label))    # Label é a classe correspondente do elemento que foi testado a distância

Após isso, o restante do código é padrão, ele armazena todas as distâncias e classes correspondentes na lista de distâncias e separa os com menor distância, e assim, há um critério de desempate ponderado baseado no inverso da distância. Na biblioteca própria do knn (sklearn.neighbors), esta métrica é chamada por "weights=distance".

Este cálculo é feito da seguinte maneira: após separados os k vizinhos mais próximos (baseando-se na distância), é adicionado, a cada um, antes do teste, o valor do inverso de sua distância mais um pequeno valor, que é 1 elevado a -5 (ou seja, 0.00001). Esse pequeno valor é adicionado para garantir que não haja divisão por zero. Vejamos o código:

    #Ordena as listas de distancias e labels pelas distancias
    DISTANCIAS.sort(key=lambda x: x[0])

    #Obtem os k vizinhos mais próx
    k_vizinhos = DISTANCIAS[:k]

    #Realiza o cálculo mencionado acima
    for distancia, label in k_vizinhos:
        contagem[label] += 1 / (distancia + 1e-5)

Todo esse cálculo dá vantagem às distâncias menores, assim ficando implícito o caráter de desempate.

## Considerações finais sobre a implementação própria:
Durante os testes do código, foi possível observas dois problemas comuns nos treinamentos de modelos de machine learning: o Underfitting e o Overfitting.

O Underfitting está relacionado a um modelo simples que não tem a precisão de capturar os padrões de treinamento, e resulta em um desempenho ruim tanto em treinamento como em teste. No knn, é relacionado com o valor de k muito alto para um conjunto de dados relativamente pequeno comparado.

Já o Overfitting ocorre quando o modelo é complexo, e se ajusta exatamente aos dados de treinamento, capturando até os ruídos possíveis, resultando em um ótimo treinamento, mas um péssimo desempeno nos dados de teste. É comumente associado no knn com um valor de k muito pequeno relacionado a um conjunto de dados bem maior.