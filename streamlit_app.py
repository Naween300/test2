import streamlit as st
import requests
import pandas as pd
import os
import plotly.express as px
from datetime import datetime

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
       
