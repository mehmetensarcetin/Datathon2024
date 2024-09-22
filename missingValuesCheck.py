#Her Sütündaki verilerilen ne kadar boş olduğunu gösterir

import pandas as pd

filePath = 'data\\train.csv'
data = pd.read_csv(filePath)

# Her sütundaki eksik değerlerin sayısını hesaplanıyor
missingValuesPerColumn = data.isnull().sum()

# Eksik değerleri ekrana yazdırın
print(missingValuesPerColumn)
