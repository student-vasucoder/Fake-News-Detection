import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

news = ["Vance, US negotiators depart meeting with Iran, Pakistan delegation after peace talks fail"]

news_vector = vectorizer.transform(news)
prediction = model.predict(news_vector)

print("Prediction:", prediction[0])