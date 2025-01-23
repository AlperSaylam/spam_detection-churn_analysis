import pickle
from sentence_transformers import SentenceTransformer


# Load the model from the file
with open('model_spam_detection.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

while True:
    message = input("Enter a message: ")
    Encoder = SentenceTransformer('distiluse-base-multilingual-cased')
    embeddings=Encoder.encode(message)
    # Use the loaded model to make predictions
    predictions = loaded_model.predict([embeddings])
    if predictions[0] == 1:
        print("Spam")
    else:
        print("Ham")
