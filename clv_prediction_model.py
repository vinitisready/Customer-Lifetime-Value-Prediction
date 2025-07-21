# Customer Lifetime Value Prediction Model
# Using Pareto/NBD and Gamma-Gamma models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import ParetoNBDFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_prepare_data(file_path):
    """
    Load transaction data and prepare it for CLV modeling
    """
    print("Loading and preparing data...")
    
    # Load the data
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Display sample data
    print("\nSample data:")
    print(df.head())
    
    # Check for required columns
    required_columns = ['CustomerID', 'InvoiceDate', 'Quantity', 'Price']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None
    
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Calculate monetary value
    df['monetary_value'] = df['Quantity'] * df['Price']
    
    # Remove returns (negative quantities)
    df = df[df['Quantity'] > 0]
    
    # Remove missing CustomerIDs
    df = df.dropna(subset=['CustomerID'])
    
    # Convert CustomerID to integer
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    return df

def create_rfm_summary(transaction_data, observation_period_end=None):
    """
    Create RFM (Recency, Frequency, Monetary) summary from transaction data
    """
    print("\nCreating RFM summary...")
    
    if observation_period_end is None:
        observation_period_end = transaction_data['InvoiceDate'].max()
    
    # Create summary data
    rfm = summary_data_from_transaction_data(
        transaction_data,
        'CustomerID',
        'InvoiceDate',
        'monetary_value',
        observation_period_end=observation_period_end
    )
    
    print(f"Created RFM summary with {rfm.shape[0]} customers")
    print("\nRFM summary statistics:")
    print(rfm.describe())
    
    return rfm

def fit_pareto_nbd_model(rfm):
    """
    Fit Pareto/NBD model to predict customer purchases
    """
    print("\nFitting Pareto/NBD model...")
    
    # Initialize and fit the Pareto/NBD model
    pareto_model = ParetoNBDFitter(penalizer_coef=0.1)
    pareto_model.fit(rfm['frequency'], rfm['recency'], rfm['T'])
    
    # Print model parameters
    print("\nPareto/NBD model parameters:")
    print(f"r: {pareto_model.params_['r']:.4f}")
    print(f"alpha: {pareto_model.params_['alpha']:.4f}")
    print(f"s: {pareto_model.params_['s']:.4f}")
    print(f"beta: {pareto_model.params_['beta']:.4f}")
    
    return pareto_model

def fit_gamma_gamma_model(rfm):
    """
    Fit Gamma-Gamma model to predict customer monetary value
    """
    print("\nFitting Gamma-Gamma model...")
    
    # Filter out customers with no purchases or zero monetary value
    rfm_filtered = rfm[(rfm['frequency'] > 0) & (rfm['monetary_value'] > 0)]
    
    # Initialize and fit the Gamma-Gamma model
    ggf = GammaGammaFitter(penalizer_coef=0.1)
    ggf.fit(rfm_filtered['frequency'], rfm_filtered['monetary_value'])
    
    # Print model parameters
    print("\nGamma-Gamma model parameters:")
    print(f"p: {ggf.params_['p']:.4f}")
    print(f"q: {ggf.params_['q']:.4f}")
    print(f"v: {ggf.params_['v']:.4f}")
    
    return ggf, rfm_filtered

def predict_clv(pareto_model, gamma_gamma_model, rfm, time_horizon=30, discount_rate=0.01):
    """
    Predict Customer Lifetime Value
    """
    print(f"\nPredicting CLV for {time_horizon} days with {discount_rate:.2%} discount rate...")
    
    # Calculate probability of being alive
    rfm['p_alive'] = pareto_model.conditional_probability_alive(
        rfm['frequency'], rfm['recency'], rfm['T']
    )
    
    # Predict expected purchases
    rfm['predicted_purchases'] = pareto_model.conditional_expected_number_of_purchases_up_to_time(
        time_horizon, rfm['frequency'], rfm['recency'], rfm['T']
    )
    
    # For customers with purchases, predict expected average value
    rfm_with_value = rfm[(rfm['frequency'] > 0) & (rfm['monetary_value'] > 0)].copy()
    rfm_with_value['expected_avg_value'] = gamma_gamma_model.conditional_expected_average_profit(
        rfm_with_value['frequency'], rfm_with_value['monetary_value']
    )
    
    # Calculate CLV
    rfm_with_value['clv'] = gamma_gamma_model.customer_lifetime_value(
        pareto_model,
        rfm_with_value['frequency'],
        rfm_with_value['recency'],
        rfm_with_value['T'],
        rfm_with_value['monetary_value'],
        time=time_horizon,
        discount_rate=discount_rate
    )
    
    print(f"Calculated CLV for {rfm_with_value.shape[0]} customers")
    print("\nCLV statistics:")
    print(rfm_with_value['clv'].describe())
    
    return rfm_with_value

def segment_customers(clv_data, n_clusters=4):
    """
    Segment customers based on CLV and other metrics
    """
    print(f"\nSegmenting customers into {n_clusters} groups...")
    
    # Select features for clustering
    features = ['frequency', 'recency', 'T', 'monetary_value', 'clv']
    X = clv_data[features].copy()
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clv_data['segment'] = kmeans.fit_predict(X_scaled)
    
    # Map segment numbers to meaningful labels based on CLV
    segment_avg_clv = clv_data.groupby('segment')['clv'].mean().sort_values()
    segment_labels = {
        segment_avg_clv.index[0]: 'Low Value',
        segment_avg_clv.index[1]: 'Medium Value',
        segment_avg_clv.index[2]: 'High Value',
        segment_avg_clv.index[3]: 'Very High Value'
    }
    
    clv_data['segment_name'] = clv_data['segment'].map(segment_labels)
    
    # Print segment statistics
    print("\nCustomer segments:")
    segment_stats = clv_data.groupby('segment_name').agg({
        'clv': ['mean', 'min', 'max', 'count'],
        'frequency': 'mean',
        'monetary_value': 'mean',
        'p_alive': 'mean'
    })
    
    print(segment_stats)
    
    return clv_data

def visualize_results(clv_data):
    """
    Create visualizations of CLV results
    """
    print("\nCreating visualizations...")
    
    # Set up the figure
    plt.figure(figsize=(15, 10))
    
    # 1. CLV Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(clv_data['clv'], kde=True)
    plt.title('Distribution of Customer Lifetime Value')
    plt.xlabel('CLV')
    plt.ylabel('Count')
    
    # 2. CLV by Segment
    plt.subplot(2, 2, 2)
    sns.boxplot(x='segment_name', y='clv', data=clv_data)
    plt.title('CLV by Customer Segment')
    plt.xlabel('Segment')
    plt.ylabel('CLV')
    plt.xticks(rotation=45)
    
    # 3. Frequency vs Monetary Value by Segment
    plt.subplot(2, 2, 3)
    sns.scatterplot(
        x='frequency', 
        y='monetary_value', 
        hue='segment_name', 
        data=clv_data,
        alpha=0.7
    )
    plt.title('Frequency vs Monetary Value by Segment')
    plt.xlabel('Frequency')
    plt.ylabel('Monetary Value')
    
    # 4. Probability Alive vs CLV
    plt.subplot(2, 2, 4)
    sns.scatterplot(
        x='p_alive', 
        y='clv', 
        hue='segment_name', 
        data=clv_data,
        alpha=0.7
    )
    plt.title('Probability Alive vs CLV')
    plt.xlabel('Probability Alive')
    plt.ylabel('CLV')
    
    plt.tight_layout()
    plt.savefig('clv_analysis_results.png')
    print("Saved visualizations to 'clv_analysis_results.png'")
    
    # Additional visualization: Segment distribution
    plt.figure(figsize=(10, 6))
    segment_counts = clv_data['segment_name'].value_counts().sort_index()
    ax = segment_counts.plot(kind='bar')
    plt.title('Number of Customers by Segment')
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    
    # Add value labels on top of bars
    for i, v in enumerate(segment_counts):
        ax.text(i, v + 5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('customer_segments.png')
    print("Saved segment distribution to 'customer_segments.png'")

def save_results(clv_data, output_file='clv_prediction_results.csv'):
    """
    Save CLV prediction results to a CSV file
    """
    print(f"\nSaving results to {output_file}...")
    clv_data.to_csv(output_file)
    print(f"Results saved successfully to {output_file}")

def main(transaction_file, time_horizon=30, discount_rate=0.01):
    """
    Main function to run the CLV prediction pipeline
    """
    print("=" * 80)
    print("CUSTOMER LIFETIME VALUE PREDICTION")
    print("=" * 80)
    
    # Load and prepare data
    transaction_data = load_and_prepare_data(transaction_file)
    if transaction_data is None:
        return
    
    # Create RFM summary
    rfm_data = create_rfm_summary(transaction_data)
    
    # Fit Pareto/NBD model
    pareto_model = fit_pareto_nbd_model(rfm_data)
    
    # Fit Gamma-Gamma model
    gamma_model, rfm_filtered = fit_gamma_gamma_model(rfm_data)
    
    # Predict CLV
    clv_data = predict_clv(pareto_model, gamma_model, rfm_filtered, time_horizon, discount_rate)
    
    # Segment customers
    segmented_data = segment_customers(clv_data)
    
    # Visualize results
    visualize_results(segmented_data)
    
    # Save results
    save_results(segmented_data)
    
    print("\nCLV prediction completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    # Use the sample transaction data file
    transaction_file = "sample_transaction_data.csv"
    
    # Run the CLV prediction with 30-day horizon and 1% discount rate
    main(transaction_file, time_horizon=30, discount_rate=0.01)