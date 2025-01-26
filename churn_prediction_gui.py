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
        user_no_outgoing_activity_in_days_value = int(user_no_outgoing_activity_in_days.get())
        
        # New inputs for additional features
        calls_outgoing_spendings = float(calls_outgoing_spendings_var.get())
        calls_outgoing_duration = float(calls_outgoing_duration_var.get())
        last_100_calls_outgoing_duration = float(last_100_calls_outgoing_duration_var.get())
        sms_outgoing_spendings = float(sms_outgoing_spendings_var.get())

        # Calculate additional features
        call_spending_per_duration = calls_outgoing_spendings / (calls_outgoing_duration if calls_outgoing_duration > 0 else 1)  # Avoid division by zero
        sms_spending_per_count = sms_outgoing_spendings / (sms_outgoing_count if sms_outgoing_count > 0 else 1)  # Avoid division by zero
        out_going_call_ratio = calls_outgoing_duration / (last_100_calls_outgoing_duration if last_100_calls_outgoing_duration > 0 else 1)  # Avoid division by zero

        # Create DataFrame for input data
        sending_data = pd.DataFrame([{
            'month': month,
            'user_spendings': user_spendings,
            'user_lifetime': user_lifetime,
            'calls_outgoing_count': calls_outgoing_count,
            'sms_outgoing_count': sms_outgoing_count,
            'call_spending_per_duration': call_spending_per_duration,
            'user_account_balance_last': user_account_balance_last,
            'sms_spending_per_count': sms_spending_per_count,
            'out_going_call_ratio': out_going_call_ratio,
            'user_no_outgoing_activity_in_days': user_no_outgoing_activity_in_days_value
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
user_no_outgoing_activity_in_days = StringVar()
calls_outgoing_spendings_var = StringVar()  # New variable for calls_outgoing_spendings
calls_outgoing_duration_var = StringVar()    # New variable for calls_outgoing_duration
last_100_calls_outgoing_duration_var = StringVar()  # New variable for last_100_calls_outgoing_duration
sms_outgoing_spendings_var = StringVar()  # New variable for sms_outgoing_spendings

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

Label(root, text="User No Outgoing Activity in Days:").grid(row=6, column=0)
Entry(root, textvariable=user_no_outgoing_activity_in_days).grid(row=6, column=1)

# New input fields for additional features
Label(root, text="Calls Outgoing Spendings:").grid(row=7, column=0)
Entry(root, textvariable=calls_outgoing_spendings_var).grid(row=7, column=1)

Label(root, text="Calls Outgoing Duration:").grid(row=8, column=0)
Entry(root, textvariable=calls_outgoing_duration_var).grid(row=8, column=1)

Label(root, text="Last 100 Calls Outgoing Duration:").grid(row=9, column=0)
Entry(root, textvariable=last_100_calls_outgoing_duration_var).grid(row=9, column=1)

Label(root, text="SMS Outgoing Spendings:").grid(row=10, column=0)
Entry(root, textvariable=sms_outgoing_spendings_var).grid(row=10, column=1)

# Create a button to make the prediction
Button(root, text="Predict Churn", command=make_prediction).grid(row=11, columnspan=2)

# Start the GUI event loop
root.mainloop()