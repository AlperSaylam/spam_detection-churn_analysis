import pickle
from sentence_transformers import SentenceTransformer
from tkinter import Tk, Label, Entry, Button, StringVar, messagebox

# Load the model from the file
with open('model_spam_detection.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Initialize the SentenceTransformer model
encoder = SentenceTransformer('distiluse-base-multilingual-cased')

# Function to make predictions
def make_prediction():
    try:
        message = message_var.get()
        embeddings = encoder.encode(message)  # Encode the message
        predictions = loaded_model.predict([embeddings])  # Use the loaded model to make predictions
        
        # Determine the result
        result = "Spam" if predictions[0] == 1 else "Ham"
        
        # Show result in a message box
        messagebox.showinfo("Prediction Result", result)

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main window
root = Tk()
root.title("Spam Detection")

# Create StringVar for input
message_var = StringVar()

# Create and place labels and entry fields
Label(root, text="Enter a message:").grid(row=0, column=0)
Entry(root, textvariable=message_var, width=50).grid(row=0, column=1)

# Create a button to make the prediction
Button(root, text="Check Spam", command=make_prediction).grid(row=1, columnspan=2)

# Start the GUI event loop
root.mainloop()