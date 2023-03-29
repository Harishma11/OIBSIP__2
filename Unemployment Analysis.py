import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#read the file
d=pd.read_csv("C:/Users/Harishma/Downloads/archive (1)/Unemployment in India.csv")
df=pd.DataFrame(d)
#first 5 rows
print(df.head())

#last 5 rows
print(df.tail())

#column names
print(df.columns)

#size (no of rows and no of columns)
print(df.shape)

# information about the data
print(df.info())
print(df.describe())


#generate a report
#profile=ProfileReport(df)
#profile.to_file(output_file="Unemployment.html")

#finding empty rows and columns
print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())

#dependent and independent variable
X = df[[' Estimated Unemployment Rate (%)', ' Estimated Employed', ' Estimated Labour Participation Rate (%)']]
Y = df[' Estimated Employed']

#train and test the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#linear regression
lm = LinearRegression()
lm.fit(X_train, Y_train)
LinearRegression()

#predictions
predictions = lm.predict(X_test)

#data visualization
#relationship between two variables
#pairplot
sns.pairplot(df)
plt.show()

#barplot
plt.bar(df['Region'],df[' Estimated Unemployment Rate (%)'])
plt.xticks(rotation=90)
plt.xlabel("States")
plt.ylabel("Unemployment Rate")
plt.title("Unemployment according to states ")
plt.show()

#scatterplot
plt.scatter(df['Region'],df[' Estimated Unemployment Rate (%)'])
plt.xticks(rotation=90)
plt.xlabel("States")
plt.ylabel("Unemployment Rate")
plt.title("Unemployment according to region ")
plt.show()

#histogram
df.columns= ["Region","Date","Frequency",
               "Estimated Unemployment Rate (%)","Estimated Employed",
               "Estimated Labour Participation Rate (%)","Area"]
plt.title("Indian Unemployment")
sns.histplot(x='Estimated Employed', hue="Region", data=df)
plt.show()



