import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



def analise_credito(dadosBancarios, dicionarioColunasDummies):

    for column, prefix in dicionarioColunasDummies.items():

        dummies = pd.get_dummies(dadosBancarios[column], prefix= prefix)
        dadosBancarios = pd.concat([dadosBancarios, dummies], axis= 1)
        dadosBancarios = dadosBancarios.drop(column, axis= 1)

    Y = dadosBancarios['default.payment.next.month'].copy()
    X = dadosBancarios.drop('default.payment.next.month', axis= 1).copy()

    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, train_size=0.8)

    escaladorTreino = StandardScaler()
    escaladorTeste = StandardScaler()

    xTrain = pd.DataFrame(escaladorTreino.fit_transform(xTrain), columns= X.columns)
    xTest = pd.DataFrame(escaladorTeste.fit_transform(xTest), columns= X.columns)

    modelo = LogisticRegression()
    modelo.fit(xTrain, yTrain)

    prevendoDefaults = modelo.predict(xTest)

    acertos = accuracy_score(yTest.values, prevendoDefaults)

    cm = confusion_matrix(yTest.values, prevendoDefaults)

    return prevendoDefaults, acertos, cm


if __name__ == '__main__':

    dadosBancarios = pd.read_csv('dados_cartao_credito.csv')
    dadosBancarios = dadosBancarios.drop('ID', axis= 1)

    dicionarioColunasDummies = {
        'EDUCATION': 'EDU', 
        'MARRIAGE': 'MAR'
    }

    prevendoDefaults, acertos, cm = analise_credito(dadosBancarios, dicionarioColunasDummies)

    print(f'{acertos*100:.2f}%')
    print(cm)



