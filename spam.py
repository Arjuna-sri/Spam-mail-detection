from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
emails = [
    "Win a $1000 gift card now!",  
    "Congratulations! You won a free vacation",  
    "Claim your free prize today!", 
    "Earn money from home easily", 
    "Cheap medicines available here",  
    "Meeting at 3 PM, don't be late",  
    "Project deadline extended till Monday",  
    "Please find the attached report",  
    "Dinner at my place tomorrow?",  
    "Let's schedule a call for discussion",  
    "Limited time offer! Buy one get one free",  
    "Urgent: Your bank account needs verification",  
    "Hello, how are you?",  
    "Happy birthday! Hope you have a great day", 
]

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0]  


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)


def predict_email(email):
    email_vector = vectorizer.transform([email])  
    prediction = model.predict(email_vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"

while True:
    test_email = input("\nEnter an email (or type 'exit' to quit): ")
    if test_email.lower() == "exit":
        break
    print("Prediction:", predict_email(test_email))
