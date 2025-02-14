import statistics, math

class StandardScaler:
    def fit(self, X_train):
        # calculate standard Deviation
        '''
        formula = sqrt.(sum(x-mean(x))/n)
        '''
        self.columns = []
        self.standard_values = []
        self.mean_values = []
        for col in X_train:
            self.columns.append(col)

            m = statistics.mean(X_train[col])
            n = len(X_train[col])
            s = sum((X_train[col] - m) ** 2)
            
            standard_value = math.sqrt(s/n)
            self.standard_values.append(standard_value)
            self.mean_values.append(m)

        print(self.columns)
        print(self.standard_values)
        print(self.mean_values)

    def transform(self, X_values):
        # 
        self.X_values = X_values

        for i, col in enumerate(self.columns):
            for j, x in enumerate(X_values[col]):
                z = ((x - self.mean_values[i]) / self.standard_values[i])
                print(z)
                self.X_values[col][j] = z
        return self.X_values

import pandas as pd
import numpy as np

df = pd.read_csv('Social_Network_Ads.csv')
df = df.iloc[:,2:]
X = df.drop('Purchased', axis=1)
y = df['Purchased']

print(np.round(X.describe(), 1))

scaler = StandardScaler()

scaler.fit(X)
X_scaled = scaler.transform(X)

# print(X_scaled.head())

print(np.round(X_scaled.describe(), 1))