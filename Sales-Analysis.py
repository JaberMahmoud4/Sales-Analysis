# Sales Data Analysis and Visualization
# Professional implementation with comprehensive data exploration

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
import plotly.graph_objects as go
import squarify

# Configure plotting style
plt.style.use('default')
sns.set_palette("husl")

# Data Loading and Preprocessing
def load_and_preprocess_data(filepath):
    """Load sales data and perform initial preprocessing."""
    data = pd.read_csv(filepath, encoding='ISO-8859-1')
    
    # Convert order date and handle missing values
    data['ORDERDATE'] = pd.to_datetime(data['ORDERDATE'], errors='coerce')
    data.dropna(subset=['ORDERDATE'], inplace=True)
    data.set_index('ORDERDATE', inplace=True)
    
    return data

# Machine Learning - Customer Segmentation
def perform_customer_segmentation(data, n_clusters=3):
    """Apply K-Means clustering for customer segmentation."""
    features = data[['QUANTITYORDERED', 'SALES']].dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(features)
    return data

# Load and process data
data = load_and_preprocess_data('sales_data_sample.csv')
data = perform_customer_segmentation(data)

# === TIME SERIES ANALYSIS ===

def plot_monthly_sales_trend(data):
    """Display monthly sales performance over time."""
    plt.figure(figsize=(12, 6))
    
    monthly_sales = data['SALES'].resample('ME').sum()
    plt.plot(monthly_sales.index, monthly_sales.values, 
             color='#2E86AB', linewidth=2, label='Monthly Sales')
    plt.fill_between(monthly_sales.index, monthly_sales.values, 
                     color='#2E86AB', alpha=0.2)
    
    plt.title('Monthly Sales Performance Trend', fontsize=16, fontweight='bold')
    plt.xlabel('Time Period')
    plt.ylabel('Sales Revenue ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add insight annotation
    plt.annotate('Seasonal patterns evident\nwith Q4 peaks', 
                xy=(pd.to_datetime("2004-11"), monthly_sales.max()), 
                xytext=(pd.to_datetime("2004-03"), monthly_sales.max() * 0.8),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.show()

# === RELATIONSHIP ANALYSIS ===

def plot_price_sales_relationship(data):
    """Analyze relationship between price and sales by deal size."""
    plt.figure(figsize=(12, 7))
    
    sns.scatterplot(data=data, x='PRICEEACH', y='SALES', hue='DEALSIZE', 
                   palette='coolwarm', alpha=0.7, s=60)
    sns.regplot(data=data, x='PRICEEACH', y='SALES', scatter=False, 
               color='navy', line_kws={'alpha': 0.6, 'linewidth': 2})
    
    plt.title('Price vs Sales Revenue Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Unit Price ($)')
    plt.ylabel('Sales Revenue ($)')
    plt.legend(title='Deal Size', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

def plot_clustering_analysis(data):
    """Visualize customer segmentation results."""
    plt.figure(figsize=(12, 7))
    
    sns.scatterplot(data=data, x='QUANTITYORDERED', y='SALES', hue='Cluster', 
                   palette='viridis', alpha=0.7, s=60)
    sns.regplot(data=data, x='QUANTITYORDERED', y='SALES', scatter=False, 
               color='darkred', line_kws={'alpha': 0.6})
    
    plt.title('Customer Segmentation: Quantity vs Sales', fontsize=16, fontweight='bold')
    plt.xlabel('Quantity Ordered')
    plt.ylabel('Sales Revenue ($)')
    plt.legend(title='Customer Segment')
    plt.tight_layout()
    plt.show()

# === PRODUCT LINE ANALYSIS ===

def plot_product_line_facets(data):
    """Multi-dimensional analysis by product line."""
    g = sns.FacetGrid(data, col="PRODUCTLINE", hue="DEALSIZE", 
                     palette="Set2", col_wrap=3, height=4, aspect=1.2)
    g.map(plt.scatter, "PRICEEACH", "SALES", alpha=0.7, s=40)
    g.add_legend(title='Deal Size')
    g.set_axis_labels("Unit Price ($)", "Sales Revenue ($)")
    g.fig.suptitle('Sales Performance Across Product Lines', y=1.02, fontsize=16)
    plt.show()

def plot_3d_analysis(data):
    """Three-dimensional sales analysis."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(data['QUANTITYORDERED'], data['SALES'], data['PRICEEACH'], 
                        c=data['Cluster'], cmap='plasma', alpha=0.6, s=50)
    
    ax.set_xlabel('Quantity Ordered')
    ax.set_ylabel('Sales Revenue ($)')
    ax.set_zlabel('Unit Price ($)')
    ax.set_title('3D Sales Analysis with Customer Segments', fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, shrink=0.8, aspect=20)
    plt.show()

# === DISTRIBUTION ANALYSIS ===

def plot_bubble_chart(data):
    """Sales distribution with bubble visualization."""
    plt.figure(figsize=(12, 8))
    
    sizes = data['QUANTITYORDERED'] * 2  # Scale for visibility
    sns.scatterplot(data=data, x='QUANTITYORDERED', y='SALES', hue='Cluster', 
                   size=sizes, sizes=(30, 300), alpha=0.6, palette='viridis')
    
    plt.title('Sales Distribution Analysis (Bubble Chart)', fontsize=16, fontweight='bold')
    plt.xlabel('Quantity Ordered')
    plt.ylabel('Sales Revenue ($)')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

# === CATEGORICAL ANALYSIS ===

def plot_product_sales_breakdown(data):
    """Product line performance with deal size segmentation."""
    plt.figure(figsize=(14, 8))
    
    product_sales = data.groupby(['PRODUCTLINE', 'DEALSIZE'])['SALES'].sum().unstack()
    product_sales.plot(kind='bar', stacked=True, colormap='Set3', width=0.8)
    
    plt.title('Sales Performance by Product Line & Deal Size', fontsize=16, fontweight='bold')
    plt.xlabel('Product Line')
    plt.ylabel('Sales Revenue ($)')
    plt.legend(title='Deal Size', bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_geographic_analysis(data):
    """Geographic sales performance analysis."""
    plt.figure(figsize=(15, 8))
    
    country_sales = data.groupby(['COUNTRY', 'TERRITORY'])['SALES'].sum().unstack(fill_value=0)
    country_sales.plot(kind='bar', stacked=True, colormap='tab20', width=0.8)
    
    plt.title('Geographic Sales Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Country')
    plt.ylabel('Sales Revenue ($)')
    plt.legend(title='Territory', bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# === COMPOSITION ANALYSIS ===

def plot_deal_size_distribution(data):
    """Deal size composition analysis."""
    plt.figure(figsize=(10, 8))
    
    deal_counts = data['DEALSIZE'].value_counts()
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    plt.pie(deal_counts, labels=deal_counts.index, autopct='%1.1f%%', 
           startangle=90, colors=colors, textprops={'fontsize': 12})
    plt.title('Deal Size Distribution', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.show()

# === STATISTICAL ANALYSIS ===

def plot_correlation_matrix(data):
    """Correlation analysis of key metrics."""
    plt.figure(figsize=(12, 9))
    
    numeric_cols = ['QUANTITYORDERED', 'PRICEEACH', 'ORDERLINENUMBER', 
                   'SALES', 'QTR_ID', 'MONTH_ID', 'YEAR_ID', 'MSRP']
    correlation_matrix = data[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', 
               center=0, square=True, fmt='.2f')
    
    plt.title('Sales Metrics Correlation Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_sales_distribution(data):
    """Statistical distribution of sales values."""
    plt.figure(figsize=(12, 7))
    
    sns.histplot(data['SALES'], bins=50, kde=True, color='steelblue', alpha=0.7)
    
    # Add statistical markers
    mean_sales = data['SALES'].mean()
    median_sales = data['SALES'].median()
    
    plt.axvline(mean_sales, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: ${mean_sales:,.0f}')
    plt.axvline(median_sales, color='orange', linestyle='--', linewidth=2, 
               label=f'Median: ${median_sales:,.0f}')
    
    plt.title('Sales Revenue Distribution Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Sales Revenue ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# === ADVANCED VISUALIZATIONS ===

def plot_radar_chart(data):
    """Product line performance radar chart."""
    labels = data['PRODUCTLINE'].unique()
    stats = [data[data['PRODUCTLINE'] == label]['SALES'].sum() for label in labels]
    
    # Normalize for better visualization
    max_stat = max(stats)
    normalized_stats = [stat/max_stat * 100 for stat in stats]
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    normalized_stats += normalized_stats[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.plot(angles, normalized_stats, 'o-', linewidth=2, color='blue')
    ax.fill(angles, normalized_stats, color='blue', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    
    plt.title('Product Line Performance Radar', size=16, fontweight='bold', pad=20)
    plt.show()

def plot_quarterly_trends(data):
    """Quarterly performance trends with area visualization."""
    plt.figure(figsize=(12, 7))
    
    quarterly_sales = data['SALES'].resample('QE').sum()
    
    plt.fill_between(quarterly_sales.index, quarterly_sales.values, 
                    color='lightblue', alpha=0.6, label='Quarterly Sales')
    plt.plot(quarterly_sales.index, quarterly_sales.values, 
            color='navy', linewidth=3, marker='o', markersize=8)
    
    plt.title('Quarterly Sales Trends', fontsize=16, fontweight='bold')
    plt.xlabel('Quarter')
    plt.ylabel('Sales Revenue ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_sales_variability(data):
    """Sales variability analysis by product line."""
    # Remove extreme outliers for clearer visualization
    q75 = data['SALES'].quantile(0.75)
    filtered_data = data[data['SALES'] <= q75]
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=filtered_data, x='PRODUCTLINE', y='SALES', 
               palette='Set2', showfliers=True)
    
    plt.title('Sales Variability by Product Line', fontsize=16, fontweight='bold')
    plt.xlabel('Product Line')
    plt.ylabel('Sales Revenue ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_violin_analysis(data):
    """Distribution density analysis."""
    q75 = data['SALES'].quantile(0.75)
    filtered_data = data[data['SALES'] <= q75]
    
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=filtered_data, x='PRODUCTLINE', y='SALES', 
                  inner='quartile', palette='viridis')
    
    plt.title('Sales Distribution Density by Product Line', fontsize=16, fontweight='bold')
    plt.xlabel('Product Line')
    plt.ylabel('Sales Revenue ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_cluster_spider_chart(data):
    """Multi-dimensional cluster comparison."""
    cluster_sales = data.groupby(['Cluster', 'PRODUCTLINE'])['SALES'].sum().unstack(fill_value=0)
    labels = cluster_sales.columns
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    colors = ['red', 'blue', 'green']
    for i, (cluster, row) in enumerate(cluster_sales.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=f'Cluster {cluster}', color=colors[i])
        ax.fill(angles, values, alpha=0.2, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.title('Cluster Performance Comparison', size=16, fontweight='bold', pad=30)
    plt.show()

# === SPECIALIZED VISUALIZATIONS ===

def plot_donut_chart(data):
    """Enhanced deal size distribution."""
    plt.figure(figsize=(10, 8))
    
    deal_counts = data['DEALSIZE'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    wedges, texts, autotexts = plt.pie(deal_counts, labels=deal_counts.index, 
                                      autopct='%1.1f%%', startangle=90, 
                                      colors=colors, wedgeprops=dict(width=0.4))
    
    # Enhance text formatting
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.title('Deal Size Distribution', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.show()

def create_sales_gauge(data):
    """Performance gauge visualization."""
    total_sales = data['SALES'].sum()
    target_sales = 15000000  # Business target
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=total_sales,
        delta={'reference': target_sales, 'relative': True, 'valueformat': '.1%'},
        title={'text': "Sales Performance vs Target", 'font': {'size': 20}},
        gauge={'axis': {'range': [0, 20000000]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 5000000], 'color': "lightgray"},
                   {'range': [5000000, 12000000], 'color': "yellow"},
                   {'range': [12000000, 20000000], 'color': "green"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': target_sales}}
    ))
    
    fig.update_layout(height=500)
    fig.show()

def plot_country_comparison(data):
    """Top performers vs others comparison."""
    plt.figure(figsize=(14, 8))
    
    country_sales = data.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False)
    top_10 = country_sales.head(10)
    others_total = country_sales.tail(-10).sum()
    
    # Create comparison data
    comparison_data = top_10.copy()
    comparison_data['Others'] = others_total
    
    bars = plt.bar(range(len(comparison_data)), comparison_data.values, 
                  color=['skyblue'] * 10 + ['coral'])
    
    plt.title('Sales Performance: Top 10 Countries vs Others', fontsize=16, fontweight='bold')
    plt.xlabel('Countries')
    plt.ylabel('Sales Revenue ($)')
    plt.xticks(range(len(comparison_data)), comparison_data.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def plot_treemap_analysis(data):
    """Hierarchical sales visualization."""
    sales_by_product = data.groupby('PRODUCTLINE')['SALES'].sum().reset_index()
    sales_by_product = sales_by_product.sort_values('SALES', ascending=False)
    
    plt.figure(figsize=(14, 10))
    
    # Create labels with sales figures
    labels = [f"{row['PRODUCTLINE']}\n${row['SALES']:,.0f}" 
             for _, row in sales_by_product.iterrows()]
    
    squarify.plot(sizes=sales_by_product['SALES'], 
                 label=labels,
                 color=sns.color_palette("viridis", len(sales_by_product)), 
                 alpha=0.8, text_kwargs={'fontsize': 10, 'weight': 'bold'})
    
    plt.title('Sales Distribution TreeMap by Product Line', fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# === EXECUTION SECTION ===

if __name__ == "__main__":
    # Time Series Analysis
    plot_monthly_sales_trend(data)
    
    # Relationship Analysis
    plot_price_sales_relationship(data)
    plot_clustering_analysis(data)
    
    # Product Analysis
    plot_product_line_facets(data)
    plot_3d_analysis(data)
    
    # Distribution Analysis
    plot_bubble_chart(data)
    
    # Categorical Analysis
    plot_product_sales_breakdown(data)
    plot_geographic_analysis(data)
    
    # Composition Analysis
    plot_deal_size_distribution(data)
    
    # Statistical Analysis
    plot_correlation_matrix(data)
    plot_sales_distribution(data)
    
    # Advanced Visualizations
    plot_radar_chart(data)
    plot_quarterly_trends(data)
    plot_sales_variability(data)
    plot_violin_analysis(data)
    plot_cluster_spider_chart(data)
    
    # Specialized Visualizations
    plot_donut_chart(data)
    create_sales_gauge(data)
    plot_country_comparison(data)
    plot_treemap_analysis(data)
    
    print("Sales analysis complete. All visualizations generated successfully.")