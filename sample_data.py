import pandas as pd
import numpy as np
import datetime

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(n_customers=500, n_transactions=5000, output_file='sample_transaction_data.csv'):
    """
    Generate sample transaction data for CLV prediction
    
    Parameters:
    -----------
    n_customers : int
        Number of unique customers
    n_transactions : int
        Total number of transactions
    output_file : str
        Path to save the CSV file
    """
    print(f"Generating sample data with {n_customers} customers and {n_transactions} transactions...")
    
    # Generate customer IDs
    customer_ids = np.arange(10001, 10001 + n_customers)
    
    # Generate transaction dates
    start_date = datetime.datetime(2021, 1, 1)
    end_date = datetime.datetime(2022, 12, 31)
    days_range = (end_date - start_date).days
    
    # Create random transaction dates
    random_days = np.random.randint(0, days_range, n_transactions)
    transaction_dates = [start_date + datetime.timedelta(days=int(day)) for day in random_days]
    
    # Create different customer segments with different behaviors
    # High value customers: frequent purchases, high monetary value
    # Medium value customers: moderate frequency, moderate monetary value
    # Low value customers: low frequency, low monetary value
    
    # Assign customers to segments
    customer_segments = np.random.choice(
        ['high', 'medium', 'low'],
        size=n_customers,
        p=[0.2, 0.3, 0.5]  # 20% high value, 30% medium value, 50% low value
    )
    
    customer_segment_map = dict(zip(customer_ids, customer_segments))
    
    # Generate transaction data
    data = []
    
    for _ in range(n_transactions):
        # Select customer based on segment probabilities
        # High value customers are more likely to make purchases
        segment_probs = [0.5, 0.3, 0.2]  # high, medium, low
        segment = np.random.choice(['high', 'medium', 'low'], p=segment_probs)
        
        # Select a random customer from the chosen segment
        segment_customers = [cid for cid, seg in customer_segment_map.items() if seg == segment]
        customer_id = np.random.choice(segment_customers)
        
        # Generate transaction date
        transaction_date = np.random.choice(transaction_dates)
        
        # Generate quantity based on customer segment
        if segment == 'high':
            quantity = np.random.randint(3, 15)
        elif segment == 'medium':
            quantity = np.random.randint(2, 8)
        else:  # low
            quantity = np.random.randint(1, 5)
        
        # Generate price based on customer segment
        if segment == 'high':
            price = np.random.uniform(50, 200)
        elif segment == 'medium':
            price = np.random.uniform(30, 100)
        else:  # low
            price = np.random.uniform(10, 50)
        
        # Add some randomness to create realistic data
        price = round(price * np.random.uniform(0.8, 1.2), 2)
        
        # Create transaction record
        data.append({
            'CustomerID': customer_id,
            'InvoiceDate': transaction_date,
            'Quantity': quantity,
            'Price': price,
            'InvoiceNo': f"INV-{np.random.randint(10000, 99999)}",
            'StockCode': f"SKU-{np.random.randint(1000, 9999)}",
            'Description': f"Product {np.random.randint(1, 100)}"
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Sort by customer ID and date
    df = df.sort_values(['CustomerID', 'InvoiceDate'])
    
    # Calculate monetary value
    df['monetary_value'] = df['Quantity'] * df['Price']
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Sample data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Generate sample data
    generate_sample_data(output_file='sample_transaction_data.csv')