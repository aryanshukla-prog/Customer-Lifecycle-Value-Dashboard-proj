import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page layout to wide mode
st.set_page_config(
    page_title="Customer Analytics & CLV Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom Styling
st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #1E3A8A; margin-bottom: 20px; }
    .section-title { font-size: 22px; font-weight: bold; color: #2563EB; margin-top: 20px; margin-bottom: 15px; }
    
    /* Updated .card class to force dark text visibility */
    .card { 
        background-color: #F3F4F6; 
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 10px;
        color: #111827; /* <-- This forces the text to be a highly visible dark gray/black */
    }
    .card h4 {
        color: #1E3A8A !important; /* Forces headers inside cards to be a crisp blue */
        margin-top: 0px;
    }
    </style>
""", unsafe_allow_html=True)


# Load data helper function with caching
@st.cache_data
def load_data():
    try:
        # final_customer_table.csv contains both RFM and CLV columns
        df = pd.read_csv('final_customer_table.csv')
        df['Total_Score'] = df['R_Score'] + df['F_Score'] + df['M_Score']
        return df
    except FileNotFoundError:
        st.error("Error: 'final_customer_table.csv' not found. Please place it in the same directory.")
        return None

df = load_data()

if df is not None:
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🎯 Dashboard Filters")
    
    # Segment Filter
    all_segments = sorted(df['Segment'].unique().tolist())
    selected_segments = st.sidebar.multiselect("Select RFM Segments", all_segments, default=all_segments)
    
    # Cluster Filter
    all_clusters = sorted(df['Cluster_Label'].unique().tolist())
    selected_clusters = st.sidebar.multiselect("Select ML Clusters", all_clusters, default=all_clusters)
    
    # CLV Tier Filter
    all_tiers = sorted(df['clv_tier'].unique().tolist())
    selected_tiers = st.sidebar.multiselect("Select CLV Tiers", all_tiers, default=all_tiers)
    
    # Filter the DataFrame based on user selections
    filtered_df = df[
        (df['Segment'].isin(selected_segments)) &
        (df['Cluster_Label'].isin(selected_clusters)) &
        (df['clv_tier'].isin(selected_tiers))
    ]

    # App Header
    st.markdown("<div class='main-title'>📊 Customer Lifecycle & Value Dashboard</div>", unsafe_allow_html=True)
    st.markdown("This dashboard combines **RFM Heuristic Segmentation**, **Machine Learning Clustering**, and **BG-NBD & Gamma-Gamma Predictive Models** to understand customer lifetime value.")

    # --- TOP-LEVEL EXECUTIVE METRICS ---
    st.markdown("<div class='section-title'>📈 Executive Summary KPIs</div>", unsafe_allow_html=True)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric(label="Total Customers", value=f"{len(filtered_df):,}")
    with kpi2:
        st.metric(label="Total Historical Revenue", value=f"${filtered_df['Monetary'].sum():,.2f}")
    with kpi3:
        st.metric(label="Projected 12M CLV Pipeline", value=f"${filtered_df['clv_12m'].sum():,.2f}")
    with kpi4:
        st.metric(label="Avg. Probability Alive P(Alive)", value=f"{filtered_df['prob_alive'].mean() * 100:.2f}%")

    # --- TABS CREATION ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Executive Overview", 
        "🎯 RFM & Cluster Analysis", 
        "🔮 BG-NBD & Gamma-Gamma Insights", 
        "🔍 Customer Profile Lookup"
    ])

    # ==========================================
    # TAB 1: EXECUTIVE OVERVIEW
    # ==========================================
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Distribution by Segment")
            seg_counts = filtered_df['Segment'].value_counts().reset_index()
            seg_counts.columns = ['Segment', 'Count']
            seg_counts = seg_counts.sort_values(by='Count', ascending=True)
            fig_seg = px.bar(seg_counts, x='Count', y='Segment', orientation='h', 
                             color='Segment', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_seg.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_seg, use_container_width=True)
            
        with col2:
            st.subheader("Proportion of Customers by CLV Tier")
            tier_counts = filtered_df['clv_tier'].value_counts().reset_index()
            tier_counts.columns = ['CLV Tier', 'Count']
            fig_tier = px.pie(tier_counts, values='Count', names='CLV Tier', 
                              color_discrete_sequence=px.colors.qualitative.Safe, hole=0.4)
            fig_tier.update_layout(height=400)
            st.plotly_chart(fig_tier, use_container_width=True)

        st.subheader("Strategic Customer Distribution across Algorithmic Clusters")
        cluster_counts = filtered_df['Cluster_Label'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster Label', 'Count']
        fig_cluster = px.bar(cluster_counts, x='Cluster Label', y='Count', 
                             color='Cluster Label', color_discrete_sequence=px.colors.qualitative.Dark24)
        st.plotly_chart(fig_cluster, use_container_width=True)

    # ==========================================
    # TAB 2: RFM & CLUSTER DEEP DIVE
    # ==========================================
    with tab2:
        st.markdown("<div class='section-title'>🎯 Recency, Frequency, & Monetary Behaviors</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Segment Matrix Aggregations**")
            metric_choice = st.selectbox("Select Aggregation Metric", ["Monetary", "Recency", "Frequency"])
            agg_df = filtered_df.groupby('Segment')[metric_choice].mean().reset_index().sort_values(by=metric_choice, ascending=False)
            
            # Formatting the series explicitly before rendering to avoid .style dependency
            agg_df_render = agg_df.copy()
            agg_df_render[metric_choice] = agg_df_render[metric_choice].map(lambda x: f"{x:,.2f}")
            st.dataframe(agg_df_render, use_container_width=True, hide_index=True)
            
        with col2:
            st.markdown("**Recency vs Frequency Relationship**")
            fig_scatter = px.scatter(
                filtered_df, 
                x='Recency', 
                y='Frequency', 
                size='Monetary', 
                color='Cluster_Label',
                hover_name='Customer_ID',
                log_y=True,
                title="Recency vs Log(Frequency) (Bubble size represents Historical Monetary value)",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ==========================================
    # TAB 3: CLV & PREDICTIVE INSIGHTS
    # ==========================================
    with tab3:
        st.markdown("<div class='section-title'>🔮 BG-NBD Probability and Gamma-Gamma Customer Lifetime Value</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Historical Value vs. Predicted 12M CLV**")
            fig_clv_scatter = px.scatter(
                filtered_df,
                x='Monetary',
                y='clv_12m',
                color='clv_tier',
                hover_data=['Customer_ID', 'prob_alive', 'predicted_purchases_12m'],
                labels={'Monetary': 'Historical Monetary ($)', 'clv_12m': 'Predicted 12M CLV ($)'},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_clv_scatter, use_container_width=True)
            
        with col2:
            st.markdown("**Average Predicted 12-Month CLV by RFM Segment**")
            clv_seg = filtered_df.groupby('Segment')['clv_12m'].mean().reset_index().sort_values(by='clv_12m', ascending=False)
            fig_clv_bar = px.bar(clv_seg, x='clv_12m', y='Segment', orientation='h', color='clv_12m',
                                 color_continuous_scale='Viridis', labels={'clv_12m': 'Avg Predicted CLV ($)'})
            st.plotly_chart(fig_clv_bar, use_container_width=True)

        st.markdown("---")
        st.markdown("**Distribution of Customer Churn Risks:**")
        col_risk1, col_risk2 = st.columns(2)
        with col_risk1:
            fig_hist_alive = px.histogram(filtered_df, x='prob_alive', nbins=30, title="Distribution of Probability of Being Alive P(Alive)",
                                         color_discrete_sequence=['#10B981'])
            st.plotly_chart(fig_hist_alive, use_container_width=True)
        with col_risk2:
            fig_hist_purch = px.histogram(filtered_df, x='predicted_purchases_12m', nbins=30, title="Distribution of Predicted Transactions (Next 12 Months)",
                                         color_discrete_sequence=['#F59E0B'])
            st.plotly_chart(fig_hist_purch, use_container_width=True)

    # ==========================================
    # TAB 4: INDIVIDUAL CUSTOMER LOOKUP
    # ==========================================
    with tab4:
        st.markdown("<div class='section-title'>🔍 Customer Profile Deep Dive</div>", unsafe_allow_html=True)
        
        search_id = st.number_input("Enter a Customer ID to fetch details:", min_value=int(df['Customer_ID'].min()), max_value=int(df['Customer_ID'].max()), value=int(df['Customer_ID'].iloc[0]))
        
        cust_row = df[df['Customer_ID'] == search_id]
        
        if not cust_row.empty:
            c = cust_row.iloc[0]
            
            # Display Profiles in neat Metric Columns
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("RFM Segment", str(c['Segment']))
            mc2.metric("ML Cluster", str(c['Cluster_Label']))
            mc3.metric("CLV Tier", str(c['clv_tier']))
            mc4.metric("RFM Score", f"{int(c['RFM_Score'])} (Total: {int(c['Total_Score'])})")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Historical vs Predictive Side-by-Side Comparison
            prof_col1, prof_col2 = st.columns(2)
            with prof_col1:
                st.markdown("<div class='card'><h4>📜 Historical Records</h4>"
                            f"<b>Recency (Days since last order):</b> {c['Recency']} days<br>"
                            f"<b>Frequency (Number of orders):</b> {c['Frequency']}<br>"
                            f"<b>Monetary Value (Total spent):</b> ${c['Monetary']:,.2f}<br>"
                            f"<b>Average Historical Order Value:</b> ${c['avg_order_value']:,.2f}"
                            "</div>", unsafe_allow_html=True)
            with prof_col2:
                st.markdown("<div class='card'><h4>🔮 BG-NBD & Gamma-Gamma Projections</h4>"
                            f"<b>Probability of Being Active P(Alive):</b> {c['prob_alive']*100:.2f}%<br>"
                            f"<b>Predicted Purchases (Next 12M):</b> {c['predicted_purchases_12m']:.2f}<br>"
                            f"<b>Predicted 12-Month CLV:</b> ${c['clv_12m']:,.2f}"
                            "</div>", unsafe_allow_html=True)
                
            # Compare with population average
            st.markdown("### 📊 Comparison vs Store Average")
            comparison_data = {
                'Metric': ['Recency', 'Frequency', 'Monetary', 'P(Alive)', 'Predicted Purchases 12M', '12M CLV'],
                'This Customer': [c['Recency'], c['Frequency'], c['Monetary'], c['prob_alive'], c['predicted_purchases_12m'], c['clv_12m']],
                'Store Average': [df['Recency'].mean(), df['Frequency'].mean(), df['Monetary'].mean(), df['prob_alive'].mean(), df['predicted_purchases_12m'].mean(), df['clv_12m'].mean()]
            }
            comp_df = pd.DataFrame(comparison_data)
            
            # Direct text string mapping to avoid pandas .style Jinja requirements completely
            comp_df['This Customer'] = comp_df['This Customer'].apply(lambda x: f"{x:.2f}" if x <= 1 else f"{x:,.2f}")
            comp_df['Store Average'] = comp_df['Store Average'].apply(lambda x: f"{x:.2f}" if x <= 1 else f"{x:,.2f}")
            
            st.table(comp_df)
        else:
            st.warning("Customer ID not found. Please try another one.")
