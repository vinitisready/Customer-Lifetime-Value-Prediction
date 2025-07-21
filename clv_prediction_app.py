import streamlit as st
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

# Set page configuration
st.set_page_config(
    page_title="Customer Lifetime Value Prediction",
    page_icon="💰",
    layout="wide"
)

# App title and description
st.title("Customer Lifetime Value Prediction App 💰")
st.markdown("""
This app predicts the future value of your customers using the Pareto/NBD and Gamma-Gamma models.
Upload your transaction data and get insights about your customers' lifetime value!
""")

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/customer-insight.png", width=100)
st.sidebar.title("Input Parameters")

# Sidebar parameters
time_horizon = st.sidebar.slider(
    "Prediction Time Horizon (days)",
    min_value=30,
    max_value=365,
    value=90,
    step=30
)

discount_rate = st.sidebar.slider(
    "Annual Discount Rate",
    min_value=0.01,
    max_value=0.20,
    value=0.10,
    step=0.01,
    format="%.2f"
)

profit_margin = st.sidebar.slider(
    "Profit Margin",
    min_value=0.05,
    max_value=0.50,
    value=0.15,
    step=0.05,
    format="%.2f"
)

# File uploader
st.subheader("Upload Transaction Data")
st.markdown("""
Your CSV file should contain the following columns:
- **CustomerID**: Unique identifier for each customer
- **InvoiceDate**: Date of the transaction
- **Quantity**: Number of items purchased
- **Price**: Price per item

You can also include other columns, but these are required.
""")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Sample data option
if st.checkbox("Use sample data instead"):
    # Create sample data
    @st.cache_data
    def create_sample_data():
        np.random.seed(42)
        n_customers = 100
        n_transactions = 1000
        
        # Generate customer IDs
        customer_ids = np.random.randint(10000, 99999, n_customers)
        
        # Generate transaction data
        data = {
            'CustomerID': np.random.choice(customer_ids, n_transactions),
            'InvoiceDate': pd.date_range(
                start='2022-01-01', 
                end='2022-12-31', 
                periods=n_transactions
            ),
            'Quantity': np.random.randint(1, 10, n_transactions),
            'Price': np.random.uniform(10, 100, n_transactions).round(2)
        }
        
        return pd.DataFrame(data)
    
    uploaded_file = create_sample_data()
    st.success("Using sample data!")

# Main function to process data and predict CLV
def predict_clv(transaction_data, time_horizon, discount_rate, profit_margin):
    # Convert InvoiceDate to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(transaction_data['InvoiceDate']):
        transaction_data['InvoiceDate'] = pd.to_datetime(transaction_data['InvoiceDate'])
    
    # Calculate monetary value
    if 'monetary_value' not in transaction_data.columns:
        transaction_data['monetary_value'] = transaction_data['Quantity'] * transaction_data['Price']
    
    # Remove returns (negative quantities)
    transaction_data = transaction_data[transaction_data['Quantity'] > 0]
    
    # Remove missing CustomerIDs
    transaction_data = transaction_data.dropna(subset=['CustomerID'])
    
    # Convert CustomerID to string to handle different types
    transaction_data['CustomerID'] = transaction_data['CustomerID'].astype(str)
    
    # Set the observation period end date
    observation_period_end = transaction_data['InvoiceDate'].max()
    
    # Create RFM summary
    rfm = summary_data_from_transaction_data(
        transaction_data,
        'CustomerID',
        'InvoiceDate',
        'monetary_value',
        observation_period_end=observation_period_end
    )
    
    # Fit Pareto/NBD model
    pareto_model = ParetoNBDFitter(penalizer_coef=0.1)
    pareto_model.fit(rfm['frequency'], rfm['recency'], rfm['T'])
    
    # Calculate probability of being alive
    rfm['p_alive'] = pareto_model.conditional_probability_alive(
        rfm['frequency'], rfm['recency'], rfm['T']
    )
    
    # Predict expected purchases
    rfm['predicted_purchases'] = pareto_model.conditional_expected_number_of_purchases_up_to_time(
        time_horizon, rfm['frequency'], rfm['recency'], rfm['T']
    )
    
    # Filter for Gamma-Gamma model
    rfm_filtered = rfm[(rfm['frequency'] > 0) & (rfm['monetary_value'] > 0)].copy()
    
    # Fit Gamma-Gamma model
    ggf = GammaGammaFitter(penalizer_coef=0.1)
    ggf.fit(rfm_filtered['frequency'], rfm_filtered['monetary_value'])
    
    # Calculate expected average value
    rfm_filtered['expected_avg_value'] = ggf.conditional_expected_average_profit(
        rfm_filtered['frequency'], rfm_filtered['monetary_value']
    )
    
    # Calculate CLV
    rfm_filtered['clv'] = ggf.customer_lifetime_value(
        pareto_model,
        rfm_filtered['frequency'],
        rfm_filtered['recency'],
        rfm_filtered['T'],
        rfm_filtered['monetary_value'],
        time=time_horizon,
        discount_rate=discount_rate
    )
    
    # Calculate profit
    rfm_filtered['profit'] = rfm_filtered['clv'] * profit_margin
    
    # Segment customers
    features = ['frequency', 'recency', 'monetary_value', 'clv']
    X = rfm_filtered[features].copy()
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm_filtered['segment'] = kmeans.fit_predict(X_scaled)
    
    # Map segment numbers to meaningful labels based on CLV
    segment_avg_clv = rfm_filtered.groupby('segment')['clv'].mean().sort_values()
    segment_labels = {
        segment_avg_clv.index[0]: 'Low Value',
        segment_avg_clv.index[1]: 'Medium Value',
        segment_avg_clv.index[2]: 'High Value',
        segment_avg_clv.index[3]: 'Very High Value'
    }
    
    rfm_filtered['segment_name'] = rfm_filtered['segment'].map(segment_labels)
    
    return rfm_filtered

# Process data when file is uploaded
if uploaded_file is not None:
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Data loading step
    status_text.text("Loading data...")
    
    try:
        if isinstance(uploaded_file, pd.DataFrame):
            # Using sample data
            df = uploaded_file
        else:
            # Using uploaded file
            df = pd.read_csv(uploaded_file)
        
        progress_bar.progress(25)
        status_text.text("Processing data...")
        
        # Check required columns
        required_columns = ['CustomerID', 'InvoiceDate', 'Quantity', 'Price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Error: Your data is missing these required columns: {', '.join(missing_columns)}")
        else:
            # Show sample of the data
            st.subheader("Sample of your data")
            st.dataframe(df.head())
            
            # Process data
            progress_bar.progress(50)
            status_text.text("Fitting models and predicting CLV...")
            
            # Predict CLV
            results = predict_clv(df, time_horizon, discount_rate, profit_margin)
            
            progress_bar.progress(75)
            status_text.text("Creating visualizations...")
            
            # Display results
            st.subheader("Customer Lifetime Value Predictions")
            
            # Summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Customers", len(results))
                st.metric("Average CLV", f"${results['clv'].mean():.2f}")
                
            with col2:
                st.metric("Total CLV", f"${results['clv'].sum():.2f}")
                st.metric("Total Expected Profit", f"${results['profit'].sum():.2f}")
            
            # Segment distribution
            st.subheader("Customer Segments")
            segment_counts = results['segment_name'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Count']
            
            # Sort by segment value
            segment_order = ['Low Value', 'Medium Value', 'High Value', 'Very High Value']
            segment_counts['Segment'] = pd.Categorical(
                segment_counts['Segment'], 
                categories=segment_order, 
                ordered=True
            )
            segment_counts = segment_counts.sort_values('Segment')
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = sns.barplot(x='Segment', y='Count', data=segment_counts, ax=ax)
            
            # Add value labels on top of bars
            for i, bar in enumerate(bars.patches):
                bars.text(
                    bar.get_x() + bar.get_width()/2.,
                    bar.get_height() + 0.3,
                    segment_counts['Count'].iloc[i],
                    ha='center'
                )
            
            plt.title('Number of Customers by Segment')
            st.pyplot(fig)
            
            # Segment statistics
            st.subheader("Segment Statistics")
            segment_stats = results.groupby('segment_name').agg({
                'clv': ['mean', 'min', 'max', 'count'],
                'frequency': 'mean',
                'monetary_value': 'mean',
                'p_alive': 'mean',
                'profit': 'sum'
            })
            
            # Flatten multi-index columns
            segment_stats.columns = ['_'.join(col).strip() for col in segment_stats.columns.values]
            segment_stats = segment_stats.reset_index()
            
            # Rename columns for clarity
            segment_stats = segment_stats.rename(columns={
                'clv_mean': 'Avg CLV',
                'clv_min': 'Min CLV',
                'clv_max': 'Max CLV',
                'clv_count': 'Customer Count',
                'frequency_mean': 'Avg Frequency',
                'monetary_value_mean': 'Avg Order Value',
                'p_alive_mean': 'Avg Probability Alive',
                'profit_sum': 'Total Profit'
            })
            
            # Format currency columns
            for col in ['Avg CLV', 'Min CLV', 'Max CLV', 'Avg Order Value', 'Total Profit']:
                segment_stats[col] = segment_stats[col].map('${:,.2f}'.format)
            
            # Format percentage columns
            segment_stats['Avg Probability Alive'] = segment_stats['Avg Probability Alive'].map('{:.1%}'.format)
            
            # Format numeric columns
            segment_stats['Avg Frequency'] = segment_stats['Avg Frequency'].map('{:.1f}'.format)
            
            # Display the table
            st.dataframe(segment_stats.set_index('segment_name'), use_container_width=True)
            
            # Visualizations
            st.subheader("CLV Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CLV Distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(results['clv'], kde=True, ax=ax)
                plt.title('Distribution of Customer Lifetime Value')
                plt.xlabel('CLV ($)')
                plt.ylabel('Count')
                st.pyplot(fig)
            
            with col2:
                # CLV by Segment
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x='segment_name', y='clv', data=results, order=segment_order, ax=ax)
                plt.title('CLV by Customer Segment')
                plt.xlabel('Segment')
                plt.ylabel('CLV ($)')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Frequency vs Monetary Value
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    x='frequency', 
                    y='monetary_value', 
                    hue='segment_name', 
                    data=results,
                    alpha=0.7,
                    ax=ax
                )
                plt.title('Frequency vs Monetary Value by Segment')
                plt.xlabel('Purchase Frequency')
                plt.ylabel('Average Order Value ($)')
                st.pyplot(fig)
            
            with col2:
                # Probability Alive vs CLV
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    x='p_alive', 
                    y='clv', 
                    hue='segment_name', 
                    data=results,
                    alpha=0.7,
                    ax=ax
                )
                plt.title('Probability Alive vs CLV')
                plt.xlabel('Probability Alive')
                plt.ylabel('CLV ($)')
                st.pyplot(fig)
            
            # Show full results table
            st.subheader("Full Results")
            st.dataframe(results.reset_index(), use_container_width=True)
            
            # Download button
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="clv_prediction_results.csv",
                mime="text/csv",
            )
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        progress_bar.empty()
        status_text.empty()

# Instructions section
else:
    st.info("👈 Upload your transaction data or use the sample data to get started!")
    
    st.subheader("How it works")
    st.markdown("""
    This app uses two probabilistic models to predict customer lifetime value:
    
    1. **Pareto/NBD Model**: Predicts how many purchases a customer will make in the future
    2. **Gamma-Gamma Model**: Predicts how much a customer will spend per purchase
    
    By combining these models, we can estimate the total future value of each customer.
    
    The app also segments customers into four groups based on their predicted value:
    - **Low Value**: Customers with the lowest predicted CLV
    - **Medium Value**: Customers with below-average predicted CLV
    - **High Value**: Customers with above-average predicted CLV
    - **Very High Value**: Your most valuable customers
    
    ### Required Data Format
    
    Your CSV file should contain at least these columns:
    - **CustomerID**: Unique identifier for each customer
    - **InvoiceDate**: Date of the transaction
    - **Quantity**: Number of items purchased
    - **Price**: Price per item
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Built with ❤️ using Streamlit, Lifetimes, and scikit-learn</p>
    <p>© 2023 Customer Lifetime Value Prediction App</p>
</div>
""", unsafe_allow_html=True)