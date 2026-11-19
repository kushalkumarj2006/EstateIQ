import os
import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify

app = Flask(__name__)

# ==========================================
# 1. LOAD MODELS & DATA (OPTIMIZED FOR 512MB)
# ==========================================
print("📦 Loading models...")
m_dir = "models"

def load_pkl(name):
    with open(os.path.join(m_dir, name), 'rb') as f:
        return joblib.load(f)

model = load_pkl('model.pkl')
residual_std = load_pkl('residual_std.pkl')
area_stats = load_pkl('area_stats.pkl')
comp_scaler = load_pkl('comp_scaler.pkl')
comp_index = load_pkl('comp_index.pkl')
comp_features = load_pkl('knn_features.pkl')
FEATURES = load_pkl('feature_columns.pkl')

# 🚨 MEMORY OPTIMIZATION: Drop all unnecessary columns from df to save RAM
# We only need these specific columns for the comparables and zipcode percentile
raw_df = load_pkl('df.pkl')
df = raw_df[['bedrooms','bathrooms','sqft_living','grade','condition','yr_built','zipcode','price']].copy()
del raw_df  # Free up memory immediately

print("🧠 Building SHAP explainer...")
shap_explainer = shap.TreeExplainer(model)
print("✅ App ready!")

# ==========================================
# 2. CORE LOGIC
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
        area_med_price, area_med_ppsf = float(df['price'].median()), float(df['price'].median())
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
# 3. FLASK ROUTES
# ==========================================
HTML_PAGE = r'''<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width, initial-scale=1.0'>
<title>EstateIQ — Real Estate Investment Analyzer</title>
<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500&display=swap' rel='stylesheet'>
<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
  --card: rgba(255,255,255,0.04); --border: rgba(255,255,255,0.08); --text: #f1f5f9;
  --dim: #94a3b8; --indigo: #6366f1; --pink: #ec4899; --cyan: #06b6d4;
  --green: #10b981; --red: #ef4444; --amber: #f59e0b;
}
body {
  font-family: 'Inter', sans-serif; background: radial-gradient(ellipse at top, #0f172a 0%, #020617 60%);
  min-height: 100vh; color: var(--text); padding: 22px 28px; overflow-x: hidden;
}
body::before, body::after {
  content: ''; position: fixed; width: 700px; height: 700px; border-radius: 50%;
  pointer-events: none; z-index: 0; animation: float 26s ease-in-out infinite;
}
body::before { top: -250px; left: -250px; background: radial-gradient(circle, rgba(99,102,241,0.14), transparent 70%); }
body::after { bottom: -250px; right: -250px; background: radial-gradient(circle, rgba(6,182,212,0.10), transparent 70%); animation-direction: reverse; }
@keyframes float { 0%, 100% { transform: translate(0, 0); } 50% { transform: translate(70px, 50px); } }
.container { max-width: 1560px; margin: 0 auto; position: relative; z-index: 1; }
header { display: flex; align-items: center; justify-content: space-between; gap: 24px; margin-bottom: 14px; text-align: left; }
header h1 { font-size: 32px; font-weight: 900; background: linear-gradient(135deg, #a5b4fc 0%, #67e8f9 50%, #f9a8d4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; letter-spacing: -1px; line-height: 1.1; }
header p { color: var(--dim); font-size: 12.5px; margin-top: 3px; }
.badge { display: inline-block; padding: 5px 14px; background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3); color: #6ee7b7; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 1px; flex-shrink: 0; white-space: nowrap; }
.badge::before { content: '\25CF'; margin-right: 6px; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
.layout { display: block; }
.panel { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 16px 18px; backdrop-filter: blur(20px); box-shadow: 0 8px 40px rgba(0,0,0,0.3); }
.panel h2 { font-size: 15px; margin-bottom: 2px; font-weight: 700; }
.panel .subtitle { color: var(--dim); font-size: 11px; }
.form-panel { display: grid; grid-template-columns: 215px 1.15fr 1.25fr 0.95fr; gap: 20px 22px; align-items: start; }
.form-head { display: flex; flex-direction: column; align-self: stretch; }
.form-head .subtitle { margin-top: 2px; }
.form-head .btn { margin-top: auto; }
.section-label { font-size: 9.5px; font-weight: 800; color: var(--indigo); text-transform: uppercase; letter-spacing: 1.1px; margin: 14px 0 8px; padding-bottom: 5px; border-bottom: 1px solid rgba(99,102,241,0.15); }
.form-col > .section-label:first-child { margin-top: 0; }
.field { margin-bottom: 8px; }
.field label { display: block; font-size: 10px; font-weight: 600; color: var(--dim); margin-bottom: 3px; }
.field input, .field select { width: 100%; padding: 7px 9px; border-radius: 7px; border: 1px solid var(--border); background: rgba(0,0,0,0.35); color: var(--text); font-family: inherit; font-size: 12.5px; outline: none; transition: all 0.2s; }
.field input:focus, .field select:focus { border-color: var(--indigo); box-shadow: 0 0 0 3px rgba(99,102,241,0.1); }
.field select option { background: #0f172a; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
.btn { background: linear-gradient(135deg, var(--indigo), var(--cyan)); border: none; color: white; width: 100%; padding: 12px; border-radius: 9px; font-family: inherit; font-size: 13px; font-weight: 700; cursor: pointer; transition: all 0.2s; margin-top: 16px; letter-spacing: 0.3px; box-shadow: 0 4px 20px rgba(99,102,241,0.3); }
.btn:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 6px 28px rgba(99,102,241,0.5); }
.btn:disabled { opacity: 0.6; cursor: not-allowed; }
#results { display: none; }
#results.show { display: grid; grid-template-columns: 1fr 1.15fr 1fr; gap: 14px; margin-top: 14px; animation: fadeUp 0.4s ease; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
.panel-hero { grid-column: 1 / -1; display: flex; align-items: center; gap: 18px; padding: 14px 18px; }
.price-hero { flex: 0 0 300px; background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(6,182,212,0.10)); border: 1px solid rgba(99,102,241,0.25); border-radius: 12px; padding: 14px 18px; text-align: center; }
.price-hero .label { font-size: 10px; font-weight: 700; color: var(--dim); text-transform: uppercase; letter-spacing: 1.4px; }
.price-hero .price { font-size: 34px; font-weight: 900; letter-spacing: -1.2px; background: linear-gradient(135deg, #a5b4fc, #67e8f9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 3px 0; }
.price-hero .range { color: var(--dim); font-size: 11.5px; font-family: 'JetBrains Mono', monospace; }
.verdict-badge { display: inline-block; padding: 5px 14px; border-radius: 20px; font-size: 11px; font-weight: 800; letter-spacing: 1px; margin-top: 9px; }
.verdict-badge.green { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }
.verdict-badge.red { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
.verdict-badge.amber { background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); }
.hero-side { flex: 1; min-width: 0; }
.area-display { font-size: 12.5px; color: var(--dim); line-height: 1.6; }
.area-display strong { color: var(--text); }
.metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 10px; }
#investment-grid { grid-template-columns: repeat(2, 1fr); }
.metric-card { background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
.metric-card .label { font-size: 9.5px; font-weight: 700; color: var(--dim); text-transform: uppercase; letter-spacing: 0.7px; }
.metric-card .value { font-size: 17px; font-weight: 700; margin-top: 3px; font-family: 'JetBrains Mono', monospace; }
.section-title { font-size: 12px; font-weight: 700; color: var(--text); margin: 14px 0 8px; display: flex; align-items: center; gap: 8px; }
.panel > .section-title:first-child { margin-top: 0; }
.section-title::before { content: ''; width: 3px; height: 13px; background: linear-gradient(180deg, var(--indigo), var(--cyan)); border-radius: 2px; }
table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
th { text-align: left; padding: 6px 6px; color: var(--dim); font-weight: 600; font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); }
td { padding: 6px 6px; border-bottom: 1px solid rgba(255,255,255,0.04); font-family: 'JetBrains Mono', monospace; font-size: 11.5px; }
tr:hover td { background: rgba(99,102,241,0.04); }
.flags { list-style: none; }
.flags li { padding: 6px 10px; margin-bottom: 5px; background: rgba(245,158,11,0.08); border-left: 3px solid var(--amber); border-radius: 6px; font-size: 11.5px; color: #fcd34d; }
.flags li.ok { background: rgba(16,185,129,0.08); border-left-color: var(--green); color: #6ee7b7; }
canvas { max-height: 240px; }
footer { text-align: center; margin-top: 18px; color: var(--dim); font-size: 10.5px; }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 0.6s linear infinite; vertical-align: middle; margin-right: 8px; }
@keyframes spin { to { transform: rotate(360deg); } }
@media (max-width: 1250px) { .form-panel { grid-template-columns: 1fr 1fr; } .form-head { grid-column: 1 / -1; } .form-head .btn { margin-top: 12px; } #results.show { grid-template-columns: 1fr 1fr; } .panel-hero { flex-direction: column; align-items: stretch; } .price-hero { flex: none; } }
@media (max-width: 780px) { body { padding: 16px 14px; } .form-panel { grid-template-columns: 1fr; } #results.show { grid-template-columns: 1fr; } header { flex-direction: column; align-items: flex-start; gap: 10px; } }
</style>
</head>
<body>
<div class='container'>
  <header>
    <div>
      <h1>EstateIQ</h1>
      <p>Real Estate Investment Analyzer — powered by Gradient Boosting + SHAP</p>
    </div>
    <span class='badge'>LIVE MODEL</span>
  </header>

  <div class='layout'>
    <div class='panel form-panel'>
      <div class='form-head'>
        <h2>Property Details</h2>
        <p class='subtitle'>Fill in the property attributes below</p>
        <button class='btn' id='analyze-btn' onclick='analyze()'>Analyze Investment</button>
      </div>
      <div class='form-col'>
        <div class='section-label'>Size & Layout</div>
        <div class='grid-3'>
          <div class='field'><label>Bedrooms</label><input type='number' id='bedrooms' value='3'></div>
          <div class='field'><label>Bathrooms</label><input type='number' step='0.25' id='bathrooms' value='2'></div>
          <div class='field'><label>Floors</label><input type='number' step='0.5' id='floors' value='1'></div>
        </div>
        <div class='grid-2'>
          <div class='field'><label>Sqft Living</label><input type='number' id='sqft_living' value='2000'></div>
          <div class='field'><label>Sqft Lot</label><input type='number' id='sqft_lot' value='5000'></div>
        </div>
        <div class='grid-2'>
          <div class='field'><label>Sqft Above</label><input type='number' id='sqft_above' value='2000'></div>
          <div class='field'><label>Sqft Basement</label><input type='number' id='sqft_basement' value='0'></div>
        </div>
        <div class='grid-2'>
          <div class='field'><label>Sqft Living (15 neighbors)</label><input type='number' id='sqft_living15' value='2000'></div>
          <div class='field'><label>Sqft Lot (15 neighbors)</label><input type='number' id='sqft_lot15' value='5000'></div>
        </div>
      </div>
      <div class='form-col'>
        <div class='section-label'>Quality & Condition</div>
        <div class='grid-3'>
          <div class='field'><label>Grade (1-13)</label><input type='number' id='grade' value='7'></div>
          <div class='field'><label>Condition (1-5)</label><input type='number' id='condition' value='3'></div>
          <div class='field'><label>View (0-4)</label><input type='number' id='view' value='0'></div>
        </div>
        <div class='field'><label>Waterfront</label><select id='waterfront'><option value='0'>No</option><option value='1'>Yes</option></select></div>
        <div class='section-label'>Age & Renovation</div>
        <div class='grid-2'>
          <div class='field'><label>Year Built</label><input type='number' id='yr_built' value='1990'></div>
          <div class='field'><label>Year Renovated (0 if never)</label><input type='number' id='yr_renovated' value='0'></div>
        </div>
      </div>
      <div class='form-col'>
        <div class='section-label'>Location</div>
        <div class='field'><label>Zipcode</label><input type='number' id='zipcode' value='98001'></div>
        <div class='grid-2'>
          <div class='field'><label>Latitude</label><input type='number' step='0.0001' id='lat' value='47.5112'></div>
          <div class='field'><label>Longitude</label><input type='number' step='0.0001' id='long' value='-122.2570'></div>
        </div>
      </div>
    </div>

    <div id='results'>
      <div class='panel panel-hero'>
        <div class='price-hero'>
          <div class='label'>Estimated Market Value</div>
          <div class='price' id='price-display'>—</div>
          <div class='range' id='range-display'>—</div>
          <div id='verdict-display'></div>
        </div>
        <div class='hero-side'>
          <div class='metrics-grid' id='metrics-grid'></div>
          <div class='area-display' id='area-display'></div>
        </div>
      </div>
      <div class='panel'>
        <div class='section-title'>Investment Metrics</div>
        <div class='metrics-grid' id='investment-grid'></div>
        <div class='section-title'>Risk Flags</div>
        <ul class='flags' id='flags-list'></ul>
      </div>
      <div class='panel'>
        <div class='section-title'>Comparable Properties</div>
        <div style='overflow-x:auto;'>
          <table id='comparables-table'>
            <thead><tr><th>Bed</th><th>Bath</th><th>Sqft</th><th>Grade</th><th>Yr</th><th>Zip</th><th>Price</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
      <div class='panel'>
        <div class='section-title'>What Drives This Price (SHAP)</div>
        <canvas id='shap-chart'></canvas>
      </div>
    </div>
  </div>
  <footer>EstateIQ — Gradient Boosting on King County House Sales · SHAP explanations · served via Render</footer>
</div>

<script>
var shapChart = null;
function fmt(n) { return '$' + Math.round(n).toLocaleString(); }

function analyze() {
  var btn = document.getElementById('analyze-btn');
  btn.disabled = true;
  btn.innerHTML = `<span class='spinner'></span>Analyzing...`;

  var payload = {
    bedrooms: +document.getElementById('bedrooms').value,
    bathrooms: +document.getElementById('bathrooms').value,
    sqft_living: +document.getElementById('sqft_living').value,
    sqft_lot: +document.getElementById('sqft_lot').value,
    floors: +document.getElementById('floors').value,
    waterfront: +document.getElementById('waterfront').value,
    view: +document.getElementById('view').value,
    condition: +document.getElementById('condition').value,
    grade: +document.getElementById('grade').value,
    sqft_above: +document.getElementById('sqft_above').value,
    sqft_basement: +document.getElementById('sqft_basement').value,
    yr_built: +document.getElementById('yr_built').value,
    yr_renovated: +document.getElementById('yr_renovated').value,
    zipcode: +document.getElementById('zipcode').value,
    lat: +document.getElementById('lat').value,
    long: +document.getElementById('long').value,
    sqft_living15: +document.getElementById('sqft_living15').value,
    sqft_lot15: +document.getElementById('sqft_lot15').value,
  };

  fetch('/api/analyze', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  }).then(r => r.json()).then(function(d) {
    if (d.error) { alert('Error: ' + d.error); btn.disabled = false; btn.innerHTML = 'Analyze Investment'; return; }
    renderResults(d);
    btn.disabled = false;
    btn.innerHTML = 'Analyze Investment';
  }).catch(function(e) {
    alert('Error: ' + e.message);
    btn.disabled = false; btn.innerHTML = 'Analyze Investment';
  });
}

function renderResults(d) {
  document.getElementById('results').classList.add('show');
  document.getElementById('price-display').textContent = fmt(d.price);
  document.getElementById('range-display').textContent = fmt(d.range_low) + '  —  ' + fmt(d.range_high) + '  (95% CI)';
  document.getElementById('verdict-display').innerHTML = `<span class='verdict-badge ${d.verdict_color}'>${d.verdict}</span>`;
  document.getElementById('metrics-grid').innerHTML =
    card('Price / Sqft', '$' + d.price_per_sqft) +
    card('Comparable Median', fmt(d.comparable_median)) +
    card('Zipcode Percentile', d.area.percentile + '%');
  document.getElementById('area-display').innerHTML =
    `Zipcode <strong>${d.area.zipcode}</strong> · Median price <strong>${fmt(d.area.median_price)}</strong> ` +
    `· Median $/sqft <strong>$${d.area.median_ppsf}</strong> · Listings <strong>${d.area.count}</strong><br>` +
    `This property is priced <strong>${d.area.ppsf_vs_area > 0 ? '+' : ''}${d.area.ppsf_vs_area}%</strong> ` +
    `per sqft vs the zipcode median.`;

  var tbody = document.querySelector('#comparables-table tbody');
  tbody.innerHTML = '';
  d.comparables.forEach(function(c) {
    tbody.innerHTML += `<tr>
      <td>${c.bedrooms}</td><td>${c.bathrooms}</td><td>${c.sqft_living}</td>
      <td>${c.grade}</td><td>${c.yr_built}</td><td>${c.zipcode}</td>
      <td style='color:#6ee7b7;'>${fmt(c.price)}</td>
    </tr>`;
  });

  document.getElementById('investment-grid').innerHTML =
    card('Monthly Rent (est.)', fmt(d.investment.monthly_rent)) +
    card('Gross Yield', d.investment.gross_yield + '%') +
    card('Cap Rate', d.investment.cap_rate + '%') +
    card('Annual Tax (est.)', fmt(d.investment.annual_tax)) +
    card('Annual Rent', fmt(d.investment.annual_rent)) +
    card('5-Year Value', fmt(d.investment.five_yr_value));

  document.getElementById('flags-list').innerHTML = d.risk_flags.map(function(f) {
    var isOk = f.indexOf('No major') === 0;
    return `<li class='${isOk ? 'ok' : ''}'>${f}</li>`;
  }).join('');

  renderShap(d.shap);
}

function card(label, value) {
  return `<div class='metric-card'><div class='label'>${label}</div><div class='value'>${value}</div></div>`;
}

function renderShap(shapData) {
  var labels = shapData.map(function(s) { return s.feature; });
  var values = shapData.map(function(s) { return s.impact; });
  var colors = values.map(function(v) { return v > 0 ? 'rgba(16,185,129,0.7)' : 'rgba(239,68,68,0.7)'; });

  if (shapChart) shapChart.destroy();
  var ctx = document.getElementById('shap-chart').getContext('2d');
  shapChart = new Chart(ctx, {
    type: 'bar',
    data: { labels: labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 4 }] },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: function(c) { return (c.raw >= 0 ? '+' : '') + '$' + Math.round(c.raw).toLocaleString(); } } } },
      scales: {
        x: { ticks: { color: '#94a3b8', callback: function(v) { return (v >= 0 ? '+' : '') + '$' + Math.round(v/1000) + 'k'; } }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#cbd5e1', font: { size: 11 } }, grid: { display: false } }
      }
    }
  });
}
</script>
</body>
</html>'''

@app.route('/')
def home():
    return HTML_PAGE

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    payload = request.get_json() or {}
    try:
        result = analyze(payload)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Render provides a PORT environment variable. We must bind to it.
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
