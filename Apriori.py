import pandas as pd
import os

obesityDT = pd.read_csv('MSWithInsulin (1).csv')
obesityDT
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Redefinir o índice do DataFrame para iniciar em 0
obesityDT = obesityDT.reset_index(drop=True)

# Remover as colunas não necessárias para o algoritmo Apriori do DataFrame
obesityDT = obesityDT.drop(columns=['age', 'sex', 'annual_income', 'race', 'seqn', 'fname', 'lname', 'marital_status'])



# Define os valores de referência para cada variável
reference_values = {
    'WaistCirc': 90,  # Exemplo: Valor de referência para circunferência da cintura
    'BMI': 30,  # Exemplo: Valor de referência para índice de massa corporal (IMC)
    'albuminuria': 1,  # Exemplo: Valor de referência para albuminúria
    'UrAlbCr': 30,  # Exemplo: Valor de referência para relação albumina/creatinina na urina
    'UricAcid': 7,  # Exemplo: Valor de referência para ácido úrico
    'GGT': 50,  # Exemplo: Valor de referência para Gama-GT
    'ALT': 35,  # Exemplo: Valor de referência para Alanina Aminotransferase (ALT)
    'AST': 40,  # Exemplo: Valor de referência para Aspartato Aminotransferase (AST)
    'CPK': 170,  # Exemplo: Valor de referência para Creatina Quinase (CPK)
    'HOMA': 2.71,  # Exemplo: Valor de referência para HOMA-IR
    'BloodGlucose': 99,  # Exemplo: Valor de referência para glicose no sangue
    'BloodInsulin': 2.6,  # Exemplo: Valor de referência para insulina no sangue
    'HDL': 40,  # Exemplo: Valor de referência para HDL
    'Trigylcerides': 150  # Exemplo: Valor de referência para triglicérides
}

# Transforma os dados em binários com base nos valores de referência
for variable, reference_value in reference_values.items():
    obesityDT[variable] = obesityDT[variable] > reference_value
    
# Inverter a transformação, atribuindo 1 para 'True' e 0 para 'False'
obesityDT = obesityDT.apply(lambda x: (x == True).astype(int))

# Importar a biblioteca mlxtend para Apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Aplicar o algoritmo Apriori para encontrar itens frequentes com um suporte mínimo
frequent_itemsets = apriori(obesityDT, min_support=0.1, use_colnames=True)

# Gerar regras de associação com uma confiança mínima
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Converter todas as colunas do DataFrame para tipos de dados booleanos
obesityDT= obesityDT.astype(bool)

# Exibir as regras de associação
print(rules.head(5))

# Reduzir a tabela para 1000 linhas
#obesityDT = obesityDT.head(10)

# Exibir as primeiras 10 linhas do DataFrame
#print(obesityDT.head(10))