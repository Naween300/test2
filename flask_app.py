from flask import Flask, request, jsonify
import joblib
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv(dotenv_path='email.env')

app = Flask(__name__)

# Load models
category_model = joblib.load('models/stacking_classifier.pkl')  # Categorization model
expense_models = joblib.load('models/expense_predictors.pkl')    # Expense prediction model(s)
scaler = joblib.load('models/scaler.pkl')
fraud_model = joblib.load('models/fraud_detection_model.pkl')

# Load budget prediction models
budget_models = {
    'Salaries & Benefits': joblib.load('models/optimized_xgb_Salaries & Benefits_forecasting_model.pkl'),
    'Operational Costs': joblib.load('models/optimized_xgb_Operational Costs_forecasting_model.pkl'),
    'Travel': joblib.load('models/optimized_xgb_Travel_forecasting_model.pkl'),
    'Petty Cash': joblib.load('models/optimized_xgb_Petty Cash_forecasting_model.pkl'),
    'Other': joblib.load('models/optimized_xgb_Other_forecasting_model.pkl')
}

# CSV file to store transactions
transactions_file = 'data/transactions.csv'

# Email settings
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USERNAME = os.getenv('EMAIL_USER')
SMTP_PASSWORD = os.getenv('EMAIL_PASS')
RECIPIENT_EMAIL = os.getenv('EMAIL_USER')

def send_fraud_alert(transaction_details, metadata=None):
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = 'Fraud Alert: Suspicious Transaction Detected'

        body = f"""
        <h2>Fraud Alert: Suspicious Transaction Detected</h2>
        <p>The following transaction has been flagged as potentially fraudulent:</p>
        <ul>
            <li><strong>Doc Type:</strong> {transaction_details['Doc Type']}</li>
            <li><strong>Document Number:</strong> {transaction_details['Document Number']}</li>
            <li><strong>Department:</strong> {transaction_details['Department']}</li>
            <li><strong>Amount:</strong> LKR {transaction_details['Amount']}</li>
            <li><strong>Predicted Category:</strong> {transaction_details['Predicted Category']}</li>
        </ul>
        """
        if metadata:
            body += "<h3>Additional Metadata:</h3><ul>"
            for key, value in metadata.items():
                body += f"<li><strong>{key}:</strong> {value}</li>"
            body += "</ul>"

        msg.attach(MIMEText(body, 'html'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(SMTP_USERNAME, RECIPIENT_EMAIL, text)
        server.quit()
        print("Fraud alert email sent successfully.")
    except Exception as e:
        print(f"Failed to send fraud alert email: {str(e)}")

@app.route('/categorize', methods=['POST'])
def categorize():
    try:
        data = request.json
        print(f"Received data: {data}")

        if not data or 'Doc Type' not in data or 'Department' not in data or 'Amount' not in data:
            raise KeyError("Expected keys 'Doc Type', 'Department', and 'Amount' not found in the request data")

        doc_type_mapping = {'Reimbursement': 2, 'Payment': 1, 'Invoice': 0}
        department_mapping = {'IT': 2, 'Marketing': 3, 'Operations': 4, 'HR': 1, 'Finance': 0}

        doc_type_numeric = doc_type_mapping.get(data['Doc Type'], 0)
        department_numeric = department_mapping.get(data['Department'], 0)
        amount = data['Amount']

        features = [doc_type_numeric, department_numeric, amount]
        feature_names = ['Doc Type', 'Department', 'Amount']
        features_df = pd.DataFrame([features], columns=feature_names)

        features_scaled = scaler.transform(features_df)

        category_mapping = {3: 'Salaries & Benefits', 0: 'Operational Costs', 4: 'Travel', 2: 'Petty Cash', 1: 'Other'}
        predicted_category = category_model.predict(features_scaled)
        predicted_category_label = category_mapping.get(predicted_category[0], "Unknown Category")

        fraud_probability = fraud_model.predict_proba(features_scaled)[0][1]
        is_fraud = bool(fraud_probability > 0.8)

        if is_fraud:
            transaction_details = {
                'Doc Type': data['Doc Type'],
                'Document Number': data['Document Number'],
                'Department': data['Department'],
                'Amount': amount,
                'Predicted Category': predicted_category_label
            }
            metadata = {
                'User ID': '123456',
                'Timestamp': '2024-08-22 01:00:00',
                'Location': 'Colombo'
            }
            send_fraud_alert(transaction_details, metadata)

        new_transaction = pd.DataFrame({
            'Document Type': [data['Doc Type']],
            'Document Number': [data['Document Number']],
            'Department': [data['Department']],
            'Amount': [amount],
            'Predicted Category': [predicted_category_label]
        })

        if os.path.exists(transactions_file):
            new_transaction.to_csv(transactions_file, mode='a', header=False, index=False)
        else:
            new_transaction.to_csv(transactions_file, mode='w', header=True, index=False)

        return jsonify({
            'predicted_category': predicted_category_label,
            'is_fraud': is_fraud
        })

    except Exception as e:
        print(f"Error in /categorize: {str(e)}")
        return str(e), 500

@app.route('/predict_expense', methods=['GET'])
def predict_expense():
    try:
        predictions = {
            'Operational Costs': expense_models['Operational Costs'].predict([[0, 0]])[0],
            'Other': expense_models['Other'].predict([[0, 0]])[0],
            'Petty Cash': expense_models['Petty Cash'].predict([[0, 0]])[0],
            'Salaries & Benefits': expense_models['Salaries & Benefits'].predict([[0, 0]])[0],
            'Travel': expense_models['Travel'].predict([[0, 0]])[0]
        }

        return jsonify(predictions)

    except Exception as e:
        print(f"Error in /predict_expense: {str(e)}")
        return str(e), 500

@app.route('/predict_budget', methods=['POST'])
def predict_budget():
    try:
        data = request.json
        year = data['year']
        month = data['month']

        predictions = {}

        for category, model in budget_models.items():
            features = pd.DataFrame({
                'year': [year],
                'month': [month],
                'lag_1': [0], 'lag_2': [0], 'lag_3': [0], 'lag_4': [0],
                'lag_5': [0], 'lag_6': [0], 'lag_7': [0], 'lag_8': [0],
                'lag_9': [0], 'lag_10': [0], 'lag_11': [0], 'lag_12': [0]
            })
            prediction = model.predict(features)[0]
            predictions[category] = float(prediction)

        return jsonify({
            'year': year,
            'month': month,
            'predicted_budgets': predictions
        })

    except Exception as e:
        print(f"Error in /predict_budget: {str(e)}")
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
