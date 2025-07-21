# Customer Lifetime Value Prediction

This project implements a Customer Lifetime Value (CLV) prediction model using the Pareto/NBD and Gamma-Gamma models. These probabilistic models are particularly well-suited for non-contractual business settings (like retail/e-commerce) where customers can churn silently without notifying the business.

## Overview

Customer Lifetime Value (CLV) is a prediction of the total value a business can expect from a customer throughout their relationship. Understanding CLV helps businesses:

- Identify high-value customers
- Optimize marketing spend
- Improve customer retention strategies
- Make data-driven decisions about customer acquisition costs

## Models Used

1. **Pareto/NBD Model**: Predicts the expected number of future purchases a customer will make
2. **Gamma-Gamma Model**: Predicts the expected monetary value of future purchases

Together, these models provide a robust framework for CLV prediction in non-contractual settings.

## Features

- Data preprocessing for CLV modeling
- RFM (Recency, Frequency, Monetary) analysis
- Customer purchase prediction using Pareto/NBD model
- Monetary value prediction using Gamma-Gamma model
- Customer segmentation based on CLV
- Visualization of results
- Streamlit web application for easy deployment

## Project Structure

- `clv_prediction_model.py`: Core implementation of the CLV prediction models
- `clv_prediction_app.py`: Streamlit web application for interactive CLV prediction
- `requirements.txt`: Required Python packages

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

Run the CLV prediction model on your transaction data:

```bash
python clv_prediction_model.py
```

### Web Application

Launch the Streamlit web application:

```bash
streamlit run clv_prediction_app.py
```

Then open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501).

## Input Data Format

Your transaction data should be in CSV format with at least the following columns:

- `CustomerID`: Unique identifier for each customer
- `InvoiceDate`: Date of the transaction
- `Quantity`: Number of items purchased
- `Price`: Price per item

## Output

The model produces:

1. Customer-level CLV predictions
2. Customer segmentation based on CLV
3. Visualizations of the results
4. CSV export of all predictions

## Customer Segmentation

Customers are segmented into four groups based on their predicted CLV:

- **Low Value**: Customers with the lowest predicted CLV
- **Medium Value**: Customers with below-average predicted CLV
- **High Value**: Customers with above-average predicted CLV
- **Very High Value**: Your most valuable customers

## References

- Fader, P. S., Hardie, B. G., & Lee, K. L. (2005). "Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model. Marketing Science, 24(2), 275-284.
- Fader, P. S., & Hardie, B. G. (2013). The Gamma-Gamma Model of Monetary Value. 
- Lifetimes Python package: https://github.com/CamDavidsonPilon/lifetimes

## License

MIT