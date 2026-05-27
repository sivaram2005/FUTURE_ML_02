from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample tickets
tickets = [
    "System crash issue",
    "Password reset needed",
    "Payment failed",
    "Network problem"
]

labels = ["High", "Medium", "High", "Low"]

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tickets)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Test prediction
test = vectorizer.transform(["Payment issue"])
prediction = model.predict(test)

print("Predicted Priority:", prediction[0])
