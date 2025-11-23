import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dynamic Pricing Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- CUSTOM CSS (Severe Aesthetic Change: Dark Mode / Cyan Accent) ---
st.markdown("""
<style>
    /* Main container styling: Deep Slate Background */
    .main { 
        background-color: #1e293b;
        color: #f8fafc; /* Light text */
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #334155; /* Medium Slate for sidebar */
        border-right: 2px solid #06b6d4; /* Cyan border */
    }
    
    /* Metric Cards: Darker Cards on Dark Background */
    .metric-box {
        background: #334155; /* Medium Slate */
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        border-left: 5px solid #06b6d4; /* Vibrant Cyan Primary Color */
    }
    .metric-label { font-size: 0.9rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 2.2rem; color: #f8fafc; font-weight: 800; }
    .metric-delta { font-size: 1rem; font-weight: 700; }
    .positive { color: #34d399; } /* Teal Green for positive changes */
    
    /* Insight Box (Now Full Width) */
    .insight-card {
        background-color: #1e293b; /* Primary dark background for card */
        border: 1px solid #475569;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px; /* Space between insights and metrics */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-header { font-size: 1.2rem; font-weight: bold; color: #06b6d4; margin-bottom: 15px; display: flex; align-items: center; }
    .recommendation {
        background: #1e293b; 
        border-left: 4px solid #06b6d4; /* Vibrant Cyan Accent */
        padding: 12px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
        color: #f8fafc;
    }
    /* Adjusted inline styles for contrast in AI Insights */
    .recommendation[style*="#facc15"] { border-left-color: #facc15 !important; background: #2f3e53 !important; }
    .recommendation[style*="#ef4444"] { border-left-color: #ef4444 !important; background: #2f3e53 !important; }
    
    /* Pricing Cards (Now in Sidebar) */
    .price-card {
        background: #1e293b; /* Darker background in sidebar */
        padding: 15px 10px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #475569;
        margin-bottom: 10px;
    }
    .price-title { font-size: 0.8rem; color: #94a3b8; font-weight: bold; text-transform: uppercase; height: 30px; display: flex; align-items: center; justify-content: center; }
    .price-tag { font-size: 1.4rem; font-weight: 900; color: #06b6d4; margin: 5px 0; }
    .bundle-highlight {
        background: linear-gradient(135deg, #06b6d4, #22d3ee); /* Cyan Gradient */
        color: #1e293b !important;
        border: none;
        box-shadow: 0 4px 8px rgba(6, 182, 212, 0.4);
    }
    .bundle-highlight .price-title, .bundle-highlight .price-tag { 
        color: #1e293b !important; 
        text-shadow: none; 
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADING (No Change in Logic) ---
@st.cache_data
def load_data():
    """Loads the WTP data directly from the specified backend file."""
    # Using the correct file name as identified in previous steps
    FILE_NAME = "Samsung_Sankalp.csv" 
    try:
        df = pd.read_csv(FILE_NAME)
        st.success(f"Successfully loaded data from {FILE_NAME}. Optimization running automatically.")
        return df
    except FileNotFoundError:
        st.error(f"FATAL ERROR: The data file '{FILE_NAME}' was not found in the application directory. Please ensure it is uploaded.")
        st.stop()
        
# --- 2. OPTIMIZATION ENGINE (No Change) ---

def calculate_baseline(df, products):
    """Calculates revenue if we only use separate pricing (no bundle)."""
    total_rev = 0
    for prod in products:
        wtp = df[prod].values
        candidates = np.unique(wtp)
        best_r = 0
        for p in candidates:
            r = p * np.sum(wtp >= p)
            if r > best_r: best_r = r
        total_rev += best_r
    return total_rev

@st.cache_data(show_spinner=False)
def solve_pricing(df, products):
    """
    Simulates Excel Evolutionary Solver using Differential Evolution.
    Finds optimal [P1, P2, ..., Pn, BundlePrice].
    """
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)

    def objective(prices):
        indiv_prices = np.array(prices[:n_prods])
        bundle_price = prices[n_prods]

        # Logic: Customer chooses Max Surplus
        surplus_indiv = np.sum(np.maximum(wtp_matrix - indiv_prices, 0), axis=1)
        surplus_bundle = bundle_sum_values - bundle_price
        
        # Vectorized Choice
        buy_bundle = (surplus_bundle >= surplus_indiv) & (surplus_bundle >= 0)
        buy_indiv = (~buy_bundle) & (surplus_indiv > 0)
        
        # Revenue Calculation
        rev_bundle = np.sum(buy_bundle) * bundle_price
        
        # For indiv revenue, we must check which items they bought
        items_bought_mask = (wtp_matrix >= indiv_prices) & buy_indiv[:, None]
        rev_indiv = np.sum(items_bought_mask * indiv_prices)

        return -(rev_bundle + rev_indiv)

    # Set Bounds
    bounds = []
    for i in range(n_prods):
        max_w = np.max(wtp_matrix[:, i])
        bounds.append((0, max_w * 1.5))
    bounds.append((0, np.max(bundle_sum_values)))

    res = differential_evolution(objective, bounds, strategy='best1bin', maxiter=50, popsize=15, tol=0.01, seed=42)
    return res.x, -res.fun

def get_customer_breakdown(df, products, optimal_prices):
    """Generates the customer-wise decision table."""
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)
    
    indiv_prices = optimal_prices[:n_prods]
    bundle_price = optimal_prices[n_prods]
    
    rows = []
    for i in range(len(df)):
        s_indiv = np.sum(np.maximum(wtp_matrix[i] - indiv_prices, 0))
        s_bundle = bundle_sum_values[i] - bundle_price
        
        decision = "None"
        revenue = 0
        surplus = 0
        items = "-"
        
        if s_bundle >= s_indiv and s_bundle >= 0:
            decision = "Bundle"
            revenue = bundle_price
            surplus = s_bundle
            items = "All Items"
        elif s_indiv > 0:
            decision = "Individual"
            surplus = s_indiv
            bought_indices = np.where(wtp_matrix[i] >= indiv_prices)[0]
            items = ", ".join([products[k] for k in bought_indices])
            revenue = np.sum(indiv_prices[bought_indices])
            
        rows.append({
            "Customer ID": i + 1,
            "Decision": decision,
            "Items Bought": items.replace("Samsung_", "").replace("_", " "),
            "Revenue": revenue,
            "Consumer Surplus": surplus
        })
    return pd.DataFrame(rows)

def generate_demand_curve(df, products, optimal_prices):
    """Generates demand curve data by varying bundle price while keeping indiv prices fixed."""
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)
    indiv_prices = optimal_prices[:n_prods]
    
    max_val = np.max(bundle_sum_values)
    price_points = np.linspace(0, max_val, 100)
    demand = []
    
    for bp in price_points:
        surplus_indiv = np.sum(np.maximum(wtp_matrix - indiv_prices, 0), axis=1)
        surplus_bundle = bundle_sum_values - bp
        buy_bundle = (surplus_bundle >= surplus_indiv) & (surplus_bundle >= 0)
        demand.append(np.sum(buy_bundle))
        
    return pd.DataFrame({"Price": price_points, "Demand": demand})

# --- MAIN APP ---

def main():
    st.title("🚀 Dynamic Pricing Optimization Dashboard") 
    st.markdown("---")

    df = load_data()
    products = df.columns.tolist()
    
    with st.spinner("Running Differential Evolution Solver... Analyzing Customer WTPs..."):
        # Run Calculations
        baseline_rev = calculate_baseline(df, products)
        opt_prices, max_rev = solve_pricing(df, products)
        customer_df = get_customer_breakdown(df, products, opt_prices)
        
        total_surplus = customer_df['Consumer Surplus'].sum()
        uplift = ((max_rev - baseline_rev) / baseline_rev) * 100
        
        # Calculate Stats for AI Insights
        bundle_price = opt_prices[-1]
        sum_indiv_opt = np.sum(opt_prices[:-1])
        discount = ((sum_indiv_opt - bundle_price) / sum_indiv_opt) * 100
        bundle_adoption = (len(customer_df[customer_df['Decision'] == 'Bundle']) / len(df)) * 100

        # --- SIDEBAR: PRICING MIX (New Location) ---
        with st.sidebar:
            st.header("3. Optimal Pricing Mix")
            st.markdown("These are the solver-calculated prices for **maximum revenue**:")
            
            # Bundle Price (Highlighted first)
            st.markdown(f"""
            <div class="price-card bundle-highlight">
                <div class="price-title">ALL-IN BUNDLE</div>
                <div class="price-tag">₹{bundle_price:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Individual Product Prices")

            # Individual Prices
            for i, prod in enumerate(products):
                p_opt = opt_prices[i]
                clean_name = prod.replace("Samsung_", "").replace("_", " ")
                st.markdown(f"""
                <div class="price-card">
                    <div class="price-title">{clean_name}</div>
                    <div class="price-tag">₹{p_opt:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
                
        # --- MAIN PAGE TOP: AI STRATEGIC INSIGHTS (New Full-Width Location) ---
        st.subheader("1. AI Strategic Insights")
        
        strategy_text = "Volume Driver" if discount > 15 else "Premium Extraction"
        marketing_focus = "Value-for-Money" if discount > 15 else "Exclusivity & Convenience"

        # Using columns for the insights to use the full width more effectively
        i1, i2, i3 = st.columns(3)
        with i1:
            st.markdown(f"""
            <div class="insight-card">
                <div class="recommendation">
                    <strong>🎯 Pricing Strategy: {strategy_text}</strong><br>
                    The solver suggests a <strong>{discount:.1f}% discount</strong> on the bundle, 
                    making individual items act as anchors.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with i2:
            st.markdown(f"""
            <div class="insight-card">
                <div class="recommendation" style="border-left-color: #facc15;">
                    <strong>📢 Marketing Angle: {marketing_focus}</strong><br>
                    Focus marketing on "Ecosystem Savings" - saving <strong>₹{(sum_indiv_opt - bundle_price):,.0f}</strong> 
                    compared to individual items.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with i3:
            st.markdown(f"""
            <div class="insight-card">
                <div class="recommendation" style="border-left-color: #ef4444;">
                    <strong>📉 Competitor Analysis</strong><br>
                    Your optimal bundle price effectively prices each item at 
                    <strong>₹{(bundle_price/len(products)):,.0f}</strong> avg.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("---")

        # --- MAIN PAGE MIDDLE: FINANCIAL METRICS (New Location) ---
        st.subheader("2. Financial Overview")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Total Revenue (Optimized)</div>
                <div class="metric-value">₹{max_rev:,.0f}</div>
                <div class="metric-delta positive">▲ {uplift:.1f}% vs Separate</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-box" style="border-left-color: #facc15;">
                <div class="metric-label">Consumer Surplus</div>
                <div class="metric-value">₹{total_surplus:,.0f}</div>
                <div class="metric-delta" style="color:#94a3b8;">Value Retained by Users</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-box" style="border-left-color: #ef4444;">
                <div class="metric-label">Bundle Adoption</div>
                <div class="metric-value">{bundle_adoption:.0f}%</div>
                <div class="metric-delta" style="color:#94a3b8;">Conversion Rate</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("---")

        # --- MAIN PAGE BOTTOM: CHARTS & DATA (New 1:1 Column Split) ---
        st.subheader("4. Behavioral Analysis")
        c_left, c_right = st.columns([1, 1])
        
        with c_left:
            # FIX: Properly closed st.markdown string and removed placeholder tag
            st.markdown("#### Bundle Demand Sensitivity") 
            
            demand_data = generate_demand_curve(df, products, opt_prices)
            
            # NOTE: Plotly figures are customized for the dark/cyan theme
            fig = px.line(
                demand_data, x="Price", y="Demand",
                title="Projected Bundle Sales vs. Price",
                labels={"Price": "Bundle Price (₹)", "Demand": "Number of Buyers"}
            )
            
            fig.add_vline(x=bundle_price, line_dash="dash", line_color="#facc15", annotation_text="Optimal Price")
            
            # Update plot layout for dark mode
            fig.update_layout(
                height=400, 
                hovermode="x unified",
                plot_bgcolor='#334155', # Dark plot background
                paper_bgcolor='#1e293b', # Main dark background
                font=dict(color='#f8fafc') # Light text
            )
            
            fig.update_traces(line_color='#06b6d4', fill='tozeroy', fillcolor='rgba(6, 182, 212, 0.1)') # Cyan line
            
            st.plotly_chart(fig, use_container_width=True)

        with c_right:
            st.markdown("#### Customer Purchase Decisions")
            st.dataframe(
                customer_df,
                column_config={
                    "Customer ID": st.column_config.NumberColumn(format="#%d"),
                    "Revenue": st.column_config.NumberColumn(format="₹%d"),
                    "Consumer Surplus": st.column_config.ProgressColumn(
                        format="₹%d",
                        min_value=0,
                        max_value=int(customer_df['Consumer Surplus'].max()),
                        # Note: Progress bar color relies on Streamlit's internal theme
                    ),
                    "Decision": st.column_config.TextColumn(),
                },
                use_container_width=True,
                height=400,
                hide_index=True
            )

if __name__ == "__main__":
    main()
