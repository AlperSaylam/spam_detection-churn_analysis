import pandas as pd
import joblib
from tkinter import Tk, Label, Entry, Button, StringVar, messagebox

# Load the encoder, scaler, and model
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.gz')
model = joblib.load('model_churn_analysis.pkl')

# Function to make predictions
def make_prediction():
    try:
        month = int(month_var.get())
        user_spendings = float(user_spendings_var.get())
        user_lifetime = float(user_lifetime_var.get())
        calls_outgoing_count = int(calls_outgoing_count_var.get())
        sms_outgoing_count = int(sms_outgoing_count_var.get())
        user_account_balance_last = float(user_account_balance_last_var.get())

        # Create DataFrame for input data
        sending_data = pd.DataFrame([{
            'month': month,
            'user_spendings': user_spendings,
            'user_lifetime': user_lifetime,
            'calls_outgoing_count': calls_outgoing_count,
            'sms_outgoing_count': sms_outgoing_count,
            'user_account_balance_last': user_account_balance_last
        }])

        # Transform and scale the data
        encoded = encoder.transform(sending_data[['month']])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['month']))
        scaled_df = pd.DataFrame(scaler.transform(sending_data.drop(columns=['month'])), columns=sending_data.drop(columns=['month']).columns)
        scaled_and_encoded_df = pd.concat([scaled_df, encoded_df], axis=1)

        # Make prediction
        prediction = model.predict(scaled_and_encoded_df)
        result = "Churn" if prediction[0] == 1 else "Not churn"
        
        # Show result in a message box
        messagebox.showinfo("Prediction Result", result)

    except Exception as e:
        messagebox.showerror("Input Error", str(e))

# Create the main window
root = Tk()
root.title("Churn Prediction")

# Create StringVar for inputs
month_var = StringVar()
user_spendings_var = StringVar()
user_lifetime_var = StringVar()
calls_outgoing_count_var = StringVar()
sms_outgoing_count_var = StringVar()
user_account_balance_last_var = StringVar()

# Create and place labels and entry fields
Label(root, text="Month (1-12):").grid(row=0, column=0)
Entry(root, textvariable=month_var).grid(row=0, column=1)

Label(root, text="User Spendings:").grid(row=1, column=0)
Entry(root, textvariable=user_spendings_var).grid(row=1, column=1)

Label(root, text="User Lifetime:").grid(row=2, column=0)
Entry(root, textvariable=user_lifetime_var).grid(row=2, column=1)

Label(root, text="Calls Outgoing Count:").grid(row=3, column=0)
Entry(root, textvariable=calls_outgoing_count_var).grid(row=3, column=1)

Label(root, text="SMS Outgoing Count:").grid(row=4, column=0)
Entry(root, textvariable=sms_outgoing_count_var).grid(row=4, column=1)

Label(root, text="User Account Balance Last:").grid(row=5, column=0)
Entry(root, textvariable=user_account_balance_last_var).grid(row=5, column=1)

# Create a button to make the prediction
Button(root, text="Predict Churn", command=make_prediction).grid(row=6, columnspan=2)

# Start the GUI event loop
root.mainloop()