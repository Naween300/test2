import streamlit as st
import pandas as pd
import os
import plotly.express as px
from datetime import datetime
from flask import Flask, jsonify
import joblib
from threading import Thread

# ---------------------- Flask App Section ----------------------

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

def categorize(data):
    try:
        if not data or 'Doc Type' not in data or 'Department' not in data or 'Amount' not in data:
            raise KeyError("Missing keys in request data")

        doc_type_mapping = {'Reimbursement': 2, 'Payment': 1, 'Invoice': 0}
        department_mapping = {'IT': 2, 'Marketing': 3, 'Operations': 4, 'HR': 1, 'Finance': 0}

        doc_type_numeric = doc_type_mapping.get(data['Doc Type'], 0)
        department_numeric = department_mapping.get(data['Department'], 0)
        amount = data['Amount']

        features = [doc_type_numeric, department_numeric, amount]
        features_df = pd.DataFrame([features], columns=['Doc Type', 'Department', 'Amount'])

        features_scaled = scaler.transform(features_df)

        category_mapping = {3: 'Salaries & Benefits', 0: 'Operational Costs', 4: 'Travel', 2: 'Petty Cash', 1: 'Other'}
        predicted_category = category_model.predict(features_scaled)
        predicted_category_label = category_mapping.get(predicted_category[0], "Unknown Category")

        fraud_probability = fraud_model.predict_proba(features_scaled)[0][1]
        is_fraud = bool(fraud_probability > 0.8)

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

        return {
            'predicted_category': predicted_category_label,
            'is_fraud': is_fraud
        }

    except Exception as e:
        return str(e), 500

def predict_expense():
    try:
        predictions = {
            'Operational Costs': expense_models['Operational Costs'].predict([[0, 0]])[0],
            'Other': expense_models['Other'].predict([[0, 0]])[0],
            'Petty Cash': expense_models['Petty Cash'].predict([[0, 0]])[0],
            'Salaries & Benefits': expense_models['Salaries & Benefits'].predict([[0, 0]])[0],
            'Travel': expense_models['Travel'].predict([[0, 0]])[0]
        }

        return predictions

    except Exception as e:
        return str(e), 500

def predict_budget(data):
    try:
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

        return {
            'year': year,
            'month': month,
            'predicted_budgets': predictions
        }

    except Exception as e:
        return str(e), 500

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# ---------------------- Streamlit App Section ----------------------

# Function to save new transaction to a CSV file
def save_transaction(doc_type, doc_number, department, amount, predicted_category, is_fraud=False):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_transaction = pd.DataFrame({
        'Document Type': [doc_type],
        'Document Number': [doc_number],
        'Department': [department],
        'Amount': [amount],
        'Predicted Category': [predicted_category],
        'Timestamp': [current_time]
    })
    
    file_path = 'data/fraud_transactions.csv' if is_fraud else 'data/new_transactions.csv'
    
    try:
        if os.path.exists(file_path):
            new_transaction.to_csv(file_path, mode='a', header=False, index=False)
        else:
            new_transaction.to_csv(file_path, mode='w', header=True, index=False)
        st.success("Transaction saved successfully.")
    except Exception as e:
        st.error(f"Failed to save the transaction: {str(e)}")

# Streamlit UI
def run_streamlit():
    st.title('💼 Expense Management & Fraud Detection')

    # Page navigation
    page = st.sidebar.selectbox("Choose a page", ["Add New Transaction", "Predicted Expenses for Next Month", "Monthly Budget Allocation", "View Live Data", "View Fraudulent Transactions"])

    if page == "Add New Transaction":
        st.header('Add New Transaction')

        # Input fields
        doc_type = st.selectbox('Document Type', ['Invoice', 'Receipt', 'Bill'])
        doc_number = st.text_input('Document Number', placeholder='e.g., 12345')
        department = st.selectbox('Department', ['HR', 'Finance', 'IT', 'Operations'])
        amount = st.text_input('Amount', placeholder='Enter the amount')

        if st.button('Add Transaction'):
            if amount:
                try:
                    amount = float(amount)
                    
                    # Prepare the data for categorization and fraud detection
                    new_transaction = {
                        'Doc Type': doc_type,
                        'Document Number': doc_number,
                        'Department': department,
                        'Amount': amount
                    }

                    # Directly call the Flask function instead of making a request
                    result = categorize(new_transaction)
                    if isinstance(result, dict):
                        predicted_category = result['predicted_category']
                        is_fraud = result.get('is_fraud', False)

                        st.write(f"Transaction Added: {doc_type}, {doc_number}, {department}, LKR {amount}")
                        st.write(f"Predicted Category: {predicted_category}")

                        # Save the transaction with the predicted category, timestamp, and fraud status
                        save_transaction(doc_type, doc_number, department, amount, predicted_category, is_fraud)

                        if is_fraud:
                            st.warning("Warning: This transaction might be fraudulent!")
                        else:
                            st.success("This transaction is not fraudulent.")

                    else:
                        st.error(f"Error: {result}")
                except ValueError:
                    st.error("Please enter a valid amount.")
            else:
                st.error("Please fill in all fields.")

    elif page == "Predicted Expenses for Next Month":
        st.header('Predicted Expenses for Next Month')

        # Directly call the Flask function instead of making a request
        predictions = predict_expense()
        if isinstance(predictions, dict):
            # Convert predictions to a DataFrame for easier manipulation
            predictions_df = pd.DataFrame(list(predictions.items()), columns=['Category', 'Predicted Expense'])

            # Create an interactive bar plot using Plotly
            fig = px.bar(predictions_df, x='Category', y='Predicted Expense', title='Predicted Expenses for Next Month',
                         labels={'Predicted Expense': 'Predicted Expense (LKR)', 'Category': 'Category'},
                         color='Category', height=400)

            # Update layout for better appearance
            fig.update_layout(showlegend=False, xaxis_title='Category', yaxis_title='Predicted Expense (LKR)')

            # Display the Plotly figure
            st.plotly_chart(fig)
        else:
            st.error(f"Error: {predictions}")

        elif page == "Monthly Budget Allocation":
        st.header('Monthly Budget Allocation')

        # Select year and month
        year = st.selectbox('Select Year', list(range(2023, 2030)))
        month_name = st.selectbox('Select Month', 
                                  ['January', 'February', 'March', 'April', 'May', 'June', 
                                   'July', 'August', 'September', 'October', 'November', 'December'])
        month = list(range(1, 13))[['January', 'February', 'March', 'April', 'May', 'June', 
                                   'July', 'August', 'September', 'October', 'November', 'December'].index(month_name)]

        if st.button('Get Budget Allocation'):
            # Prepare the data for budget prediction
            data = {'year': year, 'month': month}

            # Directly call the Flask function instead of making a request
            result = predict_budget(data)
            if isinstance(result, dict):
                predicted_budgets = result['predicted_budgets']
                
                st.subheader(f"Predicted Budget for {month_name} {year}:")
                for category, budget in predicted_budgets.items():
                    st.write(f"{category}: LKR {budget:.2f}")
                    
                # Plotting the budget allocations
                budget_df = pd.DataFrame(list(predicted_budgets.items()), columns=['Category', 'Budget'])
                fig = px.bar(budget_df, x='Category', y='Budget', title=f'Budget Allocation for {month_name} {year}',
                             labels={'Budget': 'Predicted Budget (LKR)', 'Category': 'Category'}, color='Category', height=400)
                st.plotly_chart(fig)
            else:
                st.error(f"Error: {result}")

    elif page == "View Live Data":
        st.header('View Live Data')

        # Load the saved transactions from the CSV file
        file_path = 'data/new_transactions.csv'
        
        if os.path.exists(file_path):
            try:
                # Reading the CSV file with added safety checks
                transactions_df = pd.read_csv(file_path)

                # Ensure all column names are stripped of leading/trailing spaces and are case-insensitive
                transactions_df.columns = transactions_df.columns.str.strip()

                st.subheader("Saved Transactions")

                # Display the data in a table
                st.dataframe(transactions_df)

                # Check if the 'Predicted Category' column exists, case-insensitive
                if 'Predicted Category' not in transactions_df.columns:
                    st.error("The 'Predicted Category' column is missing in the CSV file.")
                else:
                    # Plot the data using Plotly
                    if not transactions_df.empty:
                        fig = px.bar(transactions_df, x='Predicted Category', y='Amount', color='Predicted Category',
                                     title='Live Data: Amount per Document', labels={'Amount': 'Amount (LKR)', 'Document Number': 'Document Number'})
                        st.plotly_chart(fig)
            except pd.errors.ParserError as e:
                st.error("Error reading the CSV file. Please check its structure.")
                st.error(str(e))  # Print the specific error message
        else:
            st.warning("No transactions found. Please add a transaction first.")

    elif page == "View Fraudulent Transactions":
        st.header('View Fraudulent Transactions')

        # Load the saved transactions from the CSV file
        fraud_file_path = 'data/fraud_transactions.csv'
        
        if os.path.exists(fraud_file_path):
            try:
                # Reading the CSV file with added safety checks
                fraud_df = pd.read_csv(fraud_file_path)

                # Ensure all column names are stripped of leading/trailing spaces and are case-insensitive
                fraud_df.columns = fraud_df.columns.str.strip()

                st.subheader("Fraudulent Transactions")

                # Display the data in a table
                st.dataframe(fraud_df)

                # Check if the 'Predicted Category' column exists, case-insensitive
                if 'Predicted Category' not in fraud_df.columns:
                    st.error("The 'Predicted Category' column is missing in the CSV file.")
                else:
                    # Plot the data using Plotly
                    if not fraud_df.empty:
                        fig = px.bar(fraud_df, x='Department', y='Amount', color='Department',
                                     title='Fraudulent Transactions: Amount per Document', labels={'Amount': 'Amount (LKR)', 'Document Number': 'Document Number'})
                        st.plotly_chart(fig)
            except pd.errors.ParserError as e:
                st.error("Error reading the CSV file. Please check its structure.")
                st.error(str(e)) 
        else:
            st.warning("No fraudulent transactions found.")

# ---------------------- Execution Section ----------------------

if __name__ == '__main__':
    # Run Flask in a separate thread
    flask_thread = Thread(target=run_flask)
    flask_thread.start()

    # Run Streamlit in the main thread
    run_streamlit()
