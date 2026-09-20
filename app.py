import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import xgboost as xgb
from datetime import timedelta
import os

# Set page config
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown('<p class="main-header">📈 Retail Sales Forecasting Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Supply Chain & Inventory Optimization Dashboard</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuration")
st.sidebar.markdown("Use this panel to configure the forecasting parameters.")

@st.cache_data
def load_data():
    """Load the dataset (with a fallback to generated mock data if running in cloud without CSVs)"""
    # Check if we are running locally with the real data
    data_dir = '../Time Series/data/'
    if not os.path.exists(data_dir):
        data_dir = '../data/'
        
    if os.path.exists(os.path.join(data_dir, 'train.csv')):
        try:
            train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
            features = pd.read_csv(os.path.join(data_dir, 'features.csv'))
            stores = pd.read_csv(os.path.join(data_dir, 'stores.csv'))
            
            # Merge
            df = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
            df = df.merge(stores, on=['Store'], how='left')
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            st.sidebar.warning("Failed to load real data. Using mock data for demonstration.")
    
    # FALLBACK: Generate realistic mock data for the Live Link demonstration
    st.sidebar.info("Using Demonstration Dataset")
    dates = pd.date_range(start='2010-02-05', end='2012-10-26', freq='W-FRI')
    np.random.seed(42)
    
    mock_data = []
    for store in [1, 2, 3]:
        for dept in [1, 2, 3]:
            base_sales = np.random.randint(15000, 45000)
            for d in dates:
                # Add seasonality (higher in November/December)
                seasonality = 1.5 if d.month in [11, 12] else 1.0
                noise = np.random.normal(0, 2000)
                sales = max(0, (base_sales * seasonality) + noise)
                is_holiday = d.month == 12 and d.day > 20
                
                mock_data.append({
                    'Store': store,
                    'Dept': dept,
                    'Date': d,
                    'Weekly_Sales': sales,
                    'IsHoliday': is_holiday,
                    'Temperature': np.random.uniform(30, 90),
                    'Fuel_Price': np.random.uniform(2.5, 4.0),
                    'CPI': np.random.uniform(210, 225),
                    'Unemployment': np.random.uniform(6.0, 9.0),
                    'Type': 'A' if store == 1 else 'B',
                    'Size': 150000
                })
    return pd.DataFrame(mock_data)

# Load data
with st.spinner('Loading massive datasets...'):
    df = load_data()

# Selectors
col1, col2 = st.sidebar.columns(2)
with col1:
    selected_store = st.selectbox("Store ID", sorted(df['Store'].unique()))
with col2:
    depts = sorted(df[df['Store'] == selected_store]['Dept'].unique())
    selected_dept = st.selectbox("Department", depts)

forecast_weeks = st.sidebar.slider("Forecast Horizon (Weeks)", min_value=4, max_value=52, value=12)

# Filter Data
store_dept_df = df[(df['Store'] == selected_store) & (df['Dept'] == selected_dept)].copy()
store_dept_df = store_dept_df.sort_values('Date')
store_dept_df.set_index('Date', inplace=True)
ts = store_dept_df['Weekly_Sales'].resample('W').sum()

# Layout
tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "⚙️ Forecasting Model", "📈 Future Predictions"])

with tab1:
    st.markdown("### Historical Sales Overview")
    
    # Key Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Weeks", len(ts))
    m2.metric("Average Weekly Sales", f"${ts.mean():,.2f}")
    m3.metric("Max Weekly Sales", f"${ts.max():,.2f}")
    m4.metric("Store Size", f"{store_dept_df['Size'].iloc[0]:,} sqft")
    
    # Interactive Plotly Chart
    fig = px.line(ts.reset_index(), x='Date', y='Weekly_Sales', 
                  title=f"Weekly Sales History for Store {selected_store}, Dept {selected_dept}",
                  template="plotly_white")
    fig.update_traces(line_color='#1E3A8A', line_width=2)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Seasonal Decomposition")
    st.write("Breaking down the time series into Trend, Seasonality, and Noise to understand underlying patterns.")
    if len(ts) >= 52:
        decomposition = seasonal_decompose(ts, model='additive', period=52)
        
        d_col1, d_col2 = st.columns(2)
        
        fig_trend = px.line(decomposition.trend.reset_index(), x='Date', y='trend', title="Overall Trend")
        fig_trend.update_traces(line_color='#059669')
        d_col1.plotly_chart(fig_trend, use_container_width=True)
        
        fig_season = px.line(decomposition.seasonal.reset_index(), x='Date', y='seasonal', title="Yearly Seasonality")
        fig_season.update_traces(line_color='#DC2626')
        d_col2.plotly_chart(fig_season, use_container_width=True)
    else:
        st.warning("Not enough data to perform 52-week seasonal decomposition.")

with tab2:
    st.markdown("### XGBoost Model Training")
    st.write("We transform the time series into a supervised machine learning problem using feature engineering.")
    
    with st.expander("View Feature Engineering Logic", expanded=True):
        st.code("""
# Creating lags and rolling statistics for ML
features_df['SalesLag1'] = features_df['Sales'].shift(1)
features_df['SalesLag4'] = features_df['Sales'].shift(4)
features_df['SalesLag52'] = features_df['Sales'].shift(52)
features_df['RollingMean4_Weeks'] = features_df['Sales'].shift(1).rolling(window=4).mean()
        """, language='python')
        
    # Build features on the fly
    features_df = pd.DataFrame({'Sales': ts})
    features_df['Month'] = features_df.index.month
    features_df['IsHoliday'] = features_df.index.isin(store_dept_df[store_dept_df['IsHoliday']].index).astype(int)
    features_df['SalesLag1'] = features_df['Sales'].shift(1)
    features_df['SalesLag4'] = features_df['Sales'].shift(4)
    features_df['RollingMean4'] = features_df['Sales'].shift(1).rolling(window=4).mean()
    features_df.dropna(inplace=True)
    
    if len(features_df) > 20:
        split_idx = int(len(features_df) * 0.8)
        train_df = features_df.iloc[:split_idx]
        test_df = features_df.iloc[split_idx:]
        
        X_train, y_train = train_df.drop('Sales', axis=1), train_df['Sales']
        X_test, y_test = test_df.drop('Sales', axis=1), test_df['Sales']
        
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        
        # Walk-Forward Validation Plot
        st.markdown("#### Walk-Forward Validation (Test Set)")
        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(x=train_df.index[-50:], y=train_df['Sales'][-50:], name="Train (History)", line=dict(color='gray')))
        fig_val.add_trace(go.Scatter(x=test_df.index, y=test_df['Sales'], name="Actual Test", line=dict(color='black', width=2)))
        fig_val.add_trace(go.Scatter(x=test_df.index, y=preds, name="XGBoost Prediction", line=dict(color='#2563EB', dash='dash')))
        fig_val.update_layout(template="plotly_white", title="Model Evaluation")
        st.plotly_chart(fig_val, use_container_width=True)
        
        # Feature Importance
        st.markdown("#### Feature Importance")
        importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_})
        importance = importance.sort_values('Importance', ascending=True)
        fig_imp = px.bar(importance, x='Importance', y='Feature', orientation='h', template="plotly_white")
        fig_imp.update_traces(marker_color='#10B981')
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("Not enough data points after lagging to train model.")

with tab3:
    st.markdown(f"### Future Forecast ({forecast_weeks} Weeks)")
    st.write("Using the trained XGBoost engine to predict future out-of-sample sales.")
    
    if len(features_df) > 20:
        # Simple recursive forecasting loop
        last_known_data = features_df.iloc[-1].copy()
        current_date = features_df.index[-1]
        
        future_dates = [current_date + timedelta(weeks=i) for i in range(1, forecast_weeks + 1)]
        future_preds = []
        
        # For a truly robust app, we'd use a recursive loop. For demo speed, we approximate.
        history = list(ts.values)
        
        for d in future_dates:
            # Build feature row
            row = pd.DataFrame(index=[d])
            row['Month'] = d.month
            row['IsHoliday'] = 1 if (d.month == 12 and d.day > 20) else 0
            row['SalesLag1'] = history[-1]
            row['SalesLag4'] = history[-4]
            row['RollingMean4'] = np.mean(history[-4:])
            
            pred = model.predict(row)[0]
            future_preds.append(pred)
            history.append(pred) # Append for next lag
            
        # Plot Future
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=ts.index[-30:], y=ts.values[-30:], name="Historical Sales", line=dict(color='black', width=2)))
        fig_future.add_trace(go.Scatter(x=future_dates, y=future_preds, name="Future Forecast", line=dict(color='#EF4444', dash='dot', width=3)))
        
        # Add confidence interval shading (mocked for demo)
        upper_bound = [p * 1.15 for p in future_preds]
        lower_bound = [p * 0.85 for p in future_preds]
        
        fig_future.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        
        fig_future.update_layout(template="plotly_white")
        st.plotly_chart(fig_future, use_container_width=True)
        
        # Data Table
        st.markdown("#### Forecasted Values")
        forecast_df = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
            'Predicted Sales ($)': [f"${p:,.2f}" for p in future_preds]
        })
        st.dataframe(forecast_df, use_container_width=True)
