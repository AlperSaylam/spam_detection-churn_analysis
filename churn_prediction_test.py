import pandas as pd
import joblib

encoder = joblib.load('encoder.pkl')  # Use sparse=True for a sparse matrix
scaler = joblib.load('scaler.gz')
model = joblib.load('model_churn_analysis.pkl')


while True:
    month = input("Enter the month (1-12): ")
    user_spendings = input("Enter the user spendings (number): ")
    user_lifetime = input("Enter the user lifetime (number): ")
    calls_outgoing_count = input("Enter the calls outgoing count (number): ")
    sms_outgoing_count = input("Enter the sms outgoing count (number): ")
    user_account_balance_last = input("Enter the user account balance last (number): ")

    sending_data = sending_data = pd.DataFrame([{
        'month': int(month),
        'user_spendings': user_spendings,
        'user_lifetime': user_lifetime,
        'calls_outgoing_count': calls_outgoing_count,
        'sms_outgoing_count': sms_outgoing_count,
        'user_account_balance_last': user_account_balance_last
    }])

    encoded = encoder.transform(sending_data[['month']])

    # Create a new DataFrame with the encoded columns
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['month']))        
    scaled_df = pd.DataFrame(scaler.transform(sending_data.drop(columns=['month'])), columns=sending_data.drop(columns=['month']).columns)
    scaled_and_encoded_df = pd.concat([scaled_df, encoded_df], axis=1)
    #print(scaled_and_encoded_df)
    prediction = model.predict(scaled_and_encoded_df)
    if prediction[0] == 1:
        print("Churn")
    else:
        print("Not churn")