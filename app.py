import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shap
import plotly.express as px

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="EstateIQ", page_icon="🏠", layout="wide")

st.markdown("""
<style>
    .stApp { background: radial-gradient(ellipse at top, #0f172a 0%, #020617 60%); }
    h1 {
        background: linear-gradient(135deg, #a5b4fc 0%, #67e8f9 50%, #f9a8d4 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 900; letter-spacing: -1px;
    }
    .metric-box {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px; padding: 15px; text-align: center;
    }
    .metric-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 700; }
    .metric-value { font-size: 20px; font-weight: 700; color: #f1f5f9; margin-top: 5px; }
    .verdict-green { color: #6ee7b7; background: rgba(16,185,129,0.15); padding: 5px 14px; border-radius: 20px; font-weight: 800; border: 1px solid rgba(16,185,129,0.3); }
    .verdict-red { color: #fca5a5; background: rgba(239,68,68,0.15); padding: 5px 14px; border-radius: 20px; font-weight: 800; border: 1px solid rgba(239,68,68,0.3); }
    .verdict-amber { color: #fcd34d; background: rgba(245,158,11,0.15); padding: 5px 14px; border-radius: 20px; font-weight: 800; border: 1px solid rgba(245,158,11,0.3); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODELS FROM /models FOLDER
# ==========================================
@st.cache_resource
def load_artifacts():
    m_dir = "models"
    
    def load_pkl(name):
        with open(os.path.join(m_dir, name), 'rb') as f:
            return pickle.load(f)

    model = load_pkl('model.pkl')
    residual_std = load_pkl('residual_std.pkl')
    area_stats = load_pkl('area_stats.pkl')
    comp_scaler = load_pkl('comp_scaler.pkl')
    comp_index = load_pkl('comp_index.pkl')
    comp_features = load_pkl('knn_features.pkl')
    FEATURES = load_pkl('feature_columns.pkl')
    df = load_pkl('df.pkl')
    
    # SHAP objects are notoriously version-dependent. Try loading, else rebuild.
    try:
        shap_explainer = load_pkl('shap_explainer.pkl')
    except Exception:
        shap_explainer = shap.TreeExplainer(model)
        
    return model, residual_std, area_stats, comp_scaler, comp_index, comp_features, FEATURES, df, shap_explainer

model, residual_std, area_stats, comp_scaler, comp_index, comp_features, FEATURES, df, shap_explainer = load_artifacts()

# ==========================================
# 3. CORE LOGIC
# ==========================================
def build_feature_row(p):
    yr_built = int(p.get('yr_built', 1990))
    yr_renov = int(p.get('yr_renovated', 0))
    house_age = 2025 - yr_built
    is_renov = 1 if yr_renov > 0 else 0
    yrs_since_renov = (2025 - yr_renov) if is_renov else house_age
    sqft_living = float(p.get('sqft_living', 2000))
    sqft_lot = float(p.get('sqft_lot', 5000))
    sqft_basement = float(p.get('sqft_basement', 0))
    sqft_living15 = float(p.get('sqft_living15', sqft_living))
    sqft_lot15 = float(p.get('sqft_lot15', sqft_lot))
    bedrooms = int(p.get('bedrooms', 3))
    bathrooms = float(p.get('bathrooms', 2))

    return {
        'bedrooms': bedrooms, 'bathrooms': bathrooms, 'sqft_living': sqft_living,
        'sqft_lot': sqft_lot, 'floors': float(p.get('floors', 1)),
        'waterfront': int(p.get('waterfront', 0)), 'view': int(p.get('view', 0)),
        'condition': int(p.get('condition', 3)), 'grade': int(p.get('grade', 7)),
        'sqft_above': float(p.get('sqft_above', sqft_living - sqft_basement)),
        'sqft_basement': sqft_basement, 'yr_built': yr_built, 'yr_renovated': yr_renov,
        'zipcode': int(p.get('zipcode', 98001)), 'lat': float(p.get('lat', 47.5)),
        'long': float(p.get('long', -122.2)), 'sqft_living15': sqft_living15,
        'sqft_lot15': sqft_lot15, 'house_age': house_age, 'is_renovated': is_renov,
        'yrs_since_renov': yrs_since_renov, 'total_rooms': bedrooms + bathrooms,
        'sqft_ratio': sqft_living / max(sqft_lot, 1), 'basement_ratio': sqft_basement / max(sqft_living, 1),
        'living15_ratio': sqft_living / max(sqft_living15, 1), 'lot15_ratio': sqft_lot / max(sqft_lot15, 1),
    }

def find_comparables(feats):
    row = np.array([[feats[c] for c in comp_features]], dtype=float)
    row_scaled = comp_scaler.transform(row)
    dist, idx = comp_index.kneighbors(row_scaled)
    neighbors = idx[0]
    if dist[0][0] < 1e-6: neighbors = neighbors[1:] # Drop self if present
    comps = df.iloc[neighbors[:5]].copy()
    return comps[['bedrooms','bathrooms','sqft_living','grade','condition','yr_built','zipcode','price']]

def analyze(payload):
    feats = build_feature_row(payload)
    x = np.array([[feats[c] for c in FEATURES]])

    price = float(model.predict(x)[0])
    ci_low, ci_high = price - 1.96 * residual_std, price + 1.96 * residual_std
    ppsf = price / max(feats['sqft_living'], 1)

    zc = feats['zipcode']
    az = area_stats[area_stats['zipcode'] == zc]
    if len(az):
        area_med_price, area_med_ppsf = float(az['area_median_price'].iloc[0]), float(az['area_median_ppsf'].iloc[0])
        area_count = int(az['area_count'].iloc[0])
        percentile = float((df[df['zipcode'] == zc]['price'].values < price).mean() * 100)
    else:
        area_med_price, area_med_ppsf = float(df['price'].median()), float(df['price_per_sqft'].median())
        area_count, percentile = 0, 50.0

    ppsf_vs_area = (ppsf / area_med_ppsf - 1) * 100 if area_med_ppsf else 0
    comps = find_comparables(feats)
    comp_median = float(np.median(comps['price']))

    monthly_rent = price * 0.004
    annual_rent = monthly_rent * 12
    gross_yield = (annual_rent / price) * 100
    annual_tax = price * 0.0092
    cap_rate = ((annual_rent - annual_tax) / price) * 100
    five_yr_value = price * (1.04 ** 5)

    if price < comp_median * 0.92: verdict, verdict_color = "UNDERVALUED", "green"
    elif price > comp_median * 1.08: verdict, verdict_color = "OVERVALUED", "red"
    else: verdict, verdict_color = "FAIR PRICE", "amber"

    flags = []
    if ppsf_vs_area > 25: flags.append(f"Price per sqft is {ppsf_vs_area:.0f}% above zipcode median")
    if feats['house_age'] > 50 and not feats['is_renovated']: flags.append(f"Property is {feats['house_age']} years old with no renovation")
    if feats['condition'] <= 2: flags.append("Low condition score (≤2)")
    if feats['grade'] <= 5: flags.append("Below-average construction grade")
    if feats['sqft_living'] < 1000: flags.append("Small living area (< 1000 sqft)")
    if feats['bedrooms'] >= 6: flags.append("Unusually high bedroom count — verify layout")
    if not flags: flags.append("No major risk flags detected")

    sv = shap_explainer.shap_values(x)[0]
    contributions = sorted(zip(FEATURES, sv), key=lambda t: -abs(t[1]))[:8]
    
    return {
        'price': round(price), 'range_low': round(ci_low), 'range_high': round(ci_high),
        'price_per_sqft': round(ppsf),
        'area': {'zipcode': zc, 'median_price': round(area_med_price), 'median_ppsf': round(area_med_ppsf), 'count': area_count, 'percentile': round(percentile, 1), 'ppsf_vs_area': round(ppsf_vs_area, 1)},
        'comparables': comps.to_dict(orient='records'), 'comparable_median': round(comp_median),
        'investment': {'monthly_rent': round(monthly_rent), 'annual_rent': round(annual_rent), 'gross_yield': round(gross_yield, 2), 'annual_tax': round(annual_tax), 'cap_rate': round(cap_rate, 2), 'five_yr_value': round(five_yr_value)},
        'verdict': verdict, 'verdict_color': verdict_color, 'risk_flags': flags,
        'shap': [{'feature': f, 'impact': float(round(v))} for f, v in contributions]
    }

# ==========================================
# 4. UI LAYOUT
# ==========================================
st.title("🏠 EstateIQ")
st.markdown("Real Estate Investment Analyzer — powered by Gradient Boosting + SHAP")

with st.sidebar.form("property_form"):
    st.header("Property Details")
    
    st.subheader("Size & Layout")
    col1, col2, col3 = st.columns(3)
    bedrooms = col1.number_input("Beds", value=3, step=1)
    bathrooms = col2.number_input("Baths", value=2.0, step=0.25)
    floors = col3.number_input("Floors", value=1.0, step=0.5)
    
    sqft_living = st.number_input("Sqft Living", value=2000)
    sqft_lot = st.number_input("Sqft Lot", value=5000)
    sqft_above = st.number_input("Sqft Above", value=2000)
    sqft_basement = st.number_input("Sqft Basement", value=0)
    sqft_living15 = st.number_input("Sqft Living (15 neighbors)", value=2000)
    sqft_lot15 = st.number_input("Sqft Lot (15 neighbors)", value=5000)

    st.subheader("Quality & Condition")
    grade = st.number_input("Grade (1-13)", value=7)
    condition = st.number_input("Condition (1-5)", value=3)
    view = st.number_input("View (0-4)", value=0)
    waterfront = st.selectbox("Waterfront", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.subheader("Age & Location")
    yr_built = st.number_input("Year Built", value=1990)
    yr_renovated = st.number_input("Year Renovated (0 if never)", value=0)
    zipcode = st.number_input("Zipcode", value=98001)
    lat = st.number_input("Latitude", value=47.5112, format="%.4f")
    long = st.number_input("Longitude", value=-122.2570, format="%.4f")

    submitted = st.form_submit_button("Analyze Investment", use_container_width=True)

if submitted:
    payload = {
        "bedrooms": bedrooms, "bathrooms": bathrooms, "sqft_living": sqft_living, "sqft_lot": sqft_lot,
        "floors": floors, "waterfront": waterfront, "view": view, "condition": condition, "grade": grade,
        "sqft_above": sqft_above, "sqft_basement": sqft_basement, "yr_built": yr_built, "yr_renovated": yr_renovated,
        "zipcode": zipcode, "lat": lat, "long": long, "sqft_living15": sqft_living15, "sqft_lot15": sqft_lot15
    }
    
    with st.spinner("Analyzing..."):
        d = analyze(payload)
        
    st.markdown("---")
    col1, col2, col3 = st.columns([1.5, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="metric-box" style="text-align: left;">
            <div class="metric-label">Estimated Market Value</div>
            <div class="metric-value" style="font-size: 38px; color: #a5b4fc;">${d['price']:,}</div>
            <div style="color: #94a3b8; font-size: 13px;">${d['range_low']:,} — ${d['range_high']:,} (95% CI)</div>
            <div style="margin-top: 10px;"><span class="verdict-{d['verdict_color']}">{d['verdict']}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Price / Sqft</div>
            <div class="metric-value">${d['price_per_sqft']}</div>
        </div><br>
        <div class="metric-box">
            <div class="metric-label">Comparable Median</div>
            <div class="metric-value">${d['comparable_median']:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Zipcode Percentile</div>
            <div class="metric-value">{d['area']['percentile']}%</div>
        </div><br>
        <div class="metric-box">
            <div class="metric-label">Listings in Area</div>
            <div class="metric-value">{d['area']['count']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Investment Metrics")
        inv = d['investment']
        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("Monthly Rent", f"${inv['monthly_rent']:,}")
        ic2.metric("Gross Yield", f"{inv['gross_yield']}%")
        ic3.metric("Cap Rate", f"{inv['cap_rate']}%")
        
        ic4, ic5, ic6 = st.columns(3)
        ic4.metric("Annual Tax", f"${inv['annual_tax']:,}")
        ic5.metric("Annual Rent", f"${inv['annual_rent']:,}")
        ic6.metric("5-Year Value", f"${inv['five_yr_value']:,}")
        
        st.subheader("Risk Flags")
        for flag in d['risk_flags']:
            st.success(flag) if "No major" in flag else st.warning(flag)
                
    with col_right:
        st.subheader("Comparable Properties")
        comp_df = pd.DataFrame(d['comparables'])
        comp_df.columns = ['Bed', 'Bath', 'Sqft', 'Grade', 'Cond', 'Year', 'Zip', 'Price']
        st.dataframe(comp_df.style.format({'Price': '${:,.0f}'}), use_container_width=True, hide_index=True)
        
        st.subheader("What Drives This Price (SHAP)")
        shap_df = pd.DataFrame(d['shap'])
        shap_df['Color'] = shap_df['impact'].apply(lambda x: '#10b981' if x > 0 else '#ef4444')
        
        fig = px.bar(shap_df, x='impact', y='feature', orientation='h', color='Color', color_discrete_map="identity")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1'), margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False, yaxis_title="", xaxis_title="Impact on Price ($)",
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👈 Fill in the property details in the sidebar and click **Analyze Investment**.")
