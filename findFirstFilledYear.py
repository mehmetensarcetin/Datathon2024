# Bu kod tabloda bulunan sütunların ilk olarak hanig yıl dolu olduğunu tespit eder
import pandas as pd

filePath = 'data\\train.csv'
dataFrame = pd.read_csv(filePath)

firstFilledYears = {}

for column in dataFrame.columns:
    nonNullRows = dataFrame[dataFrame[column].notnull()]
    
    if not nonNullRows.empty:
        # Bu sütunda dolu olan en erken başvuru yılı
        firstYear = nonNullRows['Basvuru Yili'].min()
        firstFilledYears[column] = firstYear

firstFilledYearsDataFrame = pd.DataFrame(firstFilledYears.items(), columns=["Column", "First Year Filled"])

print(firstFilledYearsDataFrame)
