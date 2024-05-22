import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# this line reads the data set
dataset = pd.read_csv('C:/Users/DELL/Desktop/AI/Dataset/emails.csv')

# Split dataset text and spam
# زي ما موجوده في الداتا ست
# We did know the names of the cloumns using "print(dataset.columns)" and it's type "Object"
X = dataset['text'].values
y = dataset['spam'].values

# y is the spam emails the data set اصلا contains only spam
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)  # 20%for testing; 80%training

feq = CountVectorizer()  # how many times each word appears
num_X_train = feq.fit_transform(X_train)
num_X_test = feq.transform(X_test)

# Train a Naive Bayes classifier and create our spam detector
Bayes = MultinomialNB()
Bayes.fit(num_X_train, y_train)

# predecting
predictions = Bayes.predict(num_X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy*100, "%")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(conf_matrix)
"""
[[TP | FP]
[FN | TN]]
"""

# asking the user to enter the email
new_email = [input("Enter your email please: ")]
new_email_counts = feq.transform(new_email)
prediction = Bayes.predict(new_email_counts)
print("___"*10)
if prediction[0] == 1:
    print("This email is spam")

else:
    print("This email is not spam")

print("___"*10)

"""
examples:-
spam: buy iphone for 150$
not: meeting tomorrow !
"""
