import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from streamlit_gsheets import GSheetsConnection

st.set_page_config(page_title="Used Car Price Dashboard", layout="wide")

st.sidebar.title("Data selector")

option = st.sidebar.selectbox(
    "Choose dataset",
    ["Autoscout24", "Kleinanzeigen"]
)

# conn = st.connection("gsheets_autoscout24", type=GSheetsConnection)

# @st.cache_data()
# def load_data():
#     return conn.read()

@st.cache_data
def load_data_autoscout(path):
    df = pd.read_csv(path, sep=";", encoding='utf-8-sig')
    df['brand'] = df['brand'].str.capitalize()
    df['model'] = df['model'].str.capitalize()
    df['url'] = 'https://www.autoscout24.de' + df['url']
    return df

@st.cache_data
def load_data_kleinanzeigen(path):
    df = pd.read_csv(path, sep=";", encoding='utf-8-sig')
    return df

def clean_data(df):
    df_temp = df[(df.price < df.price.quantile(0.95)) &
                (df.distance < df.distance.quantile(0.95)) &
                (df.age < df.age.quantile(0.95)) &
                (df.age >= 0)]
    return df_temp

def fair_price(dataset, brand, model, age, distance, engine_power=None):
    q = (
        (dataset["brand"] == brand) &
        (dataset["model"] == model) &
        (dataset["age"] == age) &
        (dataset["distance"] <= distance)
    )

    if engine_power is not None and "transmission_kw" in dataset.columns:
        q &= dataset["transmission_kw"] <= engine_power

    return dataset.loc[q, "price"].mean()

def visualize_price_comparison(df_comparison):
    df_pivot = df_comparison.pivot(index='age', columns='model', values='price')
    fig = go.Figure([go.Scatter(x=df_pivot.index, y=df_pivot[m], mode='lines+markers', name=m) for m in df_pivot.columns])
    fig.update_layout(width=1500, height=600, xaxis_title='Age [Years]', yaxis_title='Price [EUR]',
                      title=f'Price Comparison of {df_comparison["brand"].iloc[0]} Models')
    return fig

def fit_exponential_fair_price(df, brand, model, distance, engine_power=None, max_age=20, min_points=5):
    def exp_fn(x, a, b): return a * np.exp(-b * x)
    ages, prices = [], []
    for age in range(max_age + 1):
        p = fair_price(df, brand=brand, model=model, age=age, distance=distance, engine_power=engine_power)
        if not np.isnan(p) and p > 0: ages.append(age); prices.append(p)
    if len(prices) < min_points: return None
    popt, _ = curve_fit(exp_fn, ages, prices, p0=(prices[0], 0.15), maxfev=10_000)
    x = np.linspace(0, max_age, 200)
    return {"ages_raw": np.array(ages), "prices_raw": np.array(prices), "age_fitted": x, "price_fitted": exp_fn(x, *popt), "params": {"a": popt[0], "b": popt[1]}}

def resale_value(fit_result, sell_age):
    if fit_result is None:
        return 0
    a = fit_result['params']['a']
    b = fit_result['params']['b']
    # Relative depreciation
    reduction_factor = 0.0
    price = a * np.exp(-b * (sell_age))
    return round(price * (1 - reduction_factor))

st.title("🚗 Used Car Analysis")

if option == "Autoscout24":
    conn = st.connection("gsheets_autoscout24", type=GSheetsConnection)
    df = conn.read()
    # df = load_data_autoscout("dataset2.csv")

elif option == "Kleinanzeigen":
    conn = st.connection("gsheets_kleinanzeigen", type=GSheetsConnection)
    df = conn.read()
    # df = load_data_kleinanzeigen("kleinanzeigen_cleaned.csv")

df = clean_data(df)

HAS_POWER = "transmission_kw" in df.columns
# ------------------ EDA ------------------

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Listings", len(df))
c2.metric("Avg Price (€)", int(df.price.mean()))
c3.metric("Median Age", int(df.age.median()))
c4.metric("Median Distance (km)", int(df.distance.median()))
c5.metric("Avg Engine Power (kW)", int(df.transmission_kw.mean()) if HAS_POWER else "N/A")
c6.metric("Brands", df.brand.nunique())
c7.metric("City with most cars on sale", df.seller_city.value_counts().idxmax())


c1, c2, c3, c4 = st.columns([1,1,1,3])

with c1:
    st.subheader("Price Distribution")
    st.plotly_chart(
        go.Figure(go.Histogram(x=df.price, nbinsx=50)).update_layout(height=400),
        use_container_width=True
    )
with c2:

    st.subheader("Top Brands")
    top_brands = df.brand.value_counts().head(10)
    st.plotly_chart(
        go.Figure(go.Bar(x=top_brands.index, y=top_brands.values)).update_layout(height=400),
        use_container_width=True
    )

with c3:
    # ------------------ Fair Price Tool ------------------
    st.header("Fair Price Estimator")

    brand = st.selectbox("Brand", sorted(df.brand.unique()), key="brand_main")
    if not brand:
        models = sorted(df.model.unique())
    else:
        models = sorted(df[df.brand == brand].model.dropna().unique())
    model = st.selectbox("Model", models, key="model_main")
    age = st.slider("Age (years)", 0, int(df.age.max()), 5)
    distance = st.slider("Max Distance (km)", 0, int(df.distance.max()), 300_000, step=10_000)
    # engine_power = st.slider("Max Engine Power (kW)", 30, int(df.transmission_kw.max()), 100)
    if HAS_POWER:
        engine_power = st.slider(
            "Max Engine Power (kW)",
            int(df.transmission_kw.min()),
            int(df.transmission_kw.max()),
            int(df.transmission_kw.median())
        )
    else:
        st.info("ℹ️ Engine power filter not available for Kleinanzeigen data")
    price = fair_price(df, brand, model, age, distance, engine_power if HAS_POWER else None)
    st.success(f"💰 Estimated Fair Price: {price:,.0f} €" if not np.isnan(price) else "No comparable vehicles found")

with c4:
    # ------------------ Model Comparison ------------------
    # st.header("Model Price Comparison by Age")

    # brand_cmp = st.selectbox("Comparison Brand", sorted(df.brand.unique()), key="cmp")
    brand_cmp = brand  # Use the same brand selected above for comparison
    models_cmp = df[df.brand == brand_cmp].model.value_counts().head(10).index.tolist()

    result = []
    for year in range(0, 21):
        for m in models_cmp:
            p = fair_price(df, brand_cmp, m, year, distance, engine_power if HAS_POWER else None)
            if not np.isnan(p):
                result.append({'brand': brand_cmp, 'model': m, 'age': year, 'price': p})

    df_comparison = pd.DataFrame(result)
    if not df_comparison.empty:
        st.plotly_chart(visualize_price_comparison(df_comparison), use_container_width=True)
    else:
        st.warning("Not enough data for comparison.")

# ------------------ Exponential Fit ------------------
c1, c2, c3, c4 = st.columns([1,1,2,1])

with c1:
    brand_2 = st.selectbox("Brand", sorted(df.brand.unique()), key="brand_selector")
    if not brand_2:
        models_2 = sorted(df.model.unique())
    else:
        models_2 = sorted(df[df.brand == brand_2].model.dropna().unique())
    model_2 = st.selectbox("Model", models_2, key="model_selector")
    time = st.slider("Years of usage (years)", 0, int(df.age.max()), 5, key="time_2")
    distance_2 = st.slider("Max Distance (km)", 0, int(df.distance.max()), 300_000, step=10_000, key="distance_2")
    leasing_price = st.number_input("Leasing Price (€/month)", min_value=0, value=509, key="leasing_price")
    period = st.selectbox("Period", ["Month", "Year"], key="cost_period")

with c2:
    insurance = st.number_input("Annual Insurance (€/year)", min_value=0, value=1000, key="insurance")
    tax = st.number_input("Annual Tax (€/year)", min_value=0, value=150, key="tax")
    fuel = st.number_input("Annual Fuel Cost (€/year)", min_value=0, value=1500, key="fuel")
    maintenance = st.number_input("Annual Maintenance (€/year)", min_value=0, value=600, key="maintenance")
    initial_buy = st.number_input("Buy Price (€)", min_value=0, value=20000, key="initial_buy")
    age_of_purchase = st.number_input("Age at Purchase (years)", min_value=0, max_value=int(df.age.max()), value=3, key="age_of_purchase")
    anzahlung = st.number_input("Tilgung (€)", min_value=0.0, value= 0.2 * initial_buy, key="down_payment")
    zinsen = st.number_input("Zinsen Jahrlich (%)", min_value=0.0, value=6.5, key="interest_rate")

# Compute fit
result = fit_exponential_fair_price(df, brand_2, model_2, distance_2, engine_power=engine_power if HAS_POWER else None)
# a = result['params']['a']
# b = result['params']['b']
# st.write(a , b)
if result:
    fig = go.Figure()
    # Raw points
    fig.add_scatter(x=result["ages_raw"], y=result["prices_raw"], mode="markers", name="Observed")
    # Fitted exponential curve
    fig.add_scatter(x=result["age_fitted"], y=result["price_fitted"], mode="lines", name="Exponential fit")
    # Layout
    fig.update_layout(title=f"{brand_2} {model_2} – Price vs Age", xaxis_title="Age (years)",
                      yaxis_title="Price (€)", width=800, height=500, legend=dict(y=0.9, x=0.8))
    with c3:
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Not enough data to fit exponential curve")


# Monthly Cost Breakdown
time_range = np.arange(1, time * 12 + 1) #months

loan_amount = initial_buy - anzahlung
months = time * 12
monthly_rate = zinsen / 12 / 100
monthly_loan_payment = (loan_amount * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)) if loan_amount > 0 else 0

balance = loan_amount
interest_payments = []
principal_payments = []

for m in range(1, months + 1):
    interest_payment = balance * monthly_rate
    principal_payment = monthly_loan_payment - interest_payment
    balance -= principal_payment
    interest_payments.append(interest_payment)
    principal_payments.append(principal_payment)

df_costs_leasing = pd.DataFrame({
    "Month": time_range,
    "Leasing": leasing_price,
    "Insurance": insurance / 12,
    "Tax": tax / 12,
    "Fuel": fuel / 12,
    "Maintenance": maintenance / 12
})

df_costs_buying = pd.DataFrame({
    "Month": time_range,
    "Zinsen" : interest_payments,
    "Tilgung": principal_payments,
    "Insurance": insurance / 12,
    "Tax": tax / 12,
    "Fuel": fuel / 12,
    "Maintenance": maintenance / 12 * 1.1  # Buying maintenance is 10% higher
})


if period == "Year":
    # create year column first
    df_costs_leasing["Year"] = ((df_costs_leasing["Month"]-1)//12)+1
    df_costs_leasing = df_costs_leasing.groupby("Year").sum().reset_index()  # Month column is gone now
    df_costs_buying["Year"] = ((df_costs_buying["Month"]-1)//12)+1
    df_costs_buying = df_costs_buying.groupby("Year").sum().reset_index()  # Month column is gone now

df_costs_buying.loc[0, "Initial_Buy"] = 0  # Add initial buy cost to first month
# p = fair_price(df, brand_2, model_2, age_of_purchase, distance_2, engine_power if HAS_POWER else None)
# df_costs_buying["Initial_Buy"].iloc[-1] = -p
df_costs_buying["Initial_Buy"].iloc[-1] =  -1 * resale_value(result, age_of_purchase + time) # Sell car at the end of period for X €
df_costs_buying.loc[0, "Down_Payment"] = anzahlung

df_costs_leasing["Total"] = df_costs_leasing.drop(columns=df_costs_leasing.columns[0]).sum(axis=1)
df_costs_leasing["Total"] = df_costs_leasing["Total"].cumsum().round(2)

df_costs_buying["Total"] = df_costs_buying.drop(columns=df_costs_buying.columns[0]).sum(axis=1)
df_costs_buying["Total"] = df_costs_buying["Total"].cumsum().round(2)

with c4:
    c1, c2 = st.columns(2)
    leasing_cost = df_costs_leasing['Total'].iloc[-1]
    buying_cost = df_costs_buying['Total'].iloc[-1]
    with c1:
        st.metric("Total Leasing Cost (€)", f"{leasing_cost:,.0f} €", delta =f"{(leasing_cost/buying_cost-1)*100:.2f} % vs Buying",  \
                  delta_color="inverse")
    with c2:
        st.metric("Total Buying Cost (€)", f"{buying_cost:,.0f} €", delta=f"{(buying_cost/leasing_cost-1)*100:.2f} % vs Leasing", \
                  delta_color="inverse")
    if leasing_cost < buying_cost:
        st.success(f"Leasing is cheaper by {buying_cost - leasing_cost:,.0f} € over {time} years")
    else:
        st.success(f"Buying is cheaper by {leasing_cost - buying_cost:,.0f} € over {time} years")
    
    st.write(f"**Resale Value after {time} years:** {resale_value(result, age_of_purchase + time):,.0f} €"
             if resale_value(result, age_of_purchase + time) is not None else "**Resale Value after {time} years:** N/A")


c1, c2 = st.columns(2)

with c1:

    fig = go.Figure()
    for col in ["Leasing", "Fuel","Insurance","Tax","Maintenance"]:
        fig.add_bar(x=df_costs_leasing[period], y=df_costs_leasing[col], name=col)
    fig.add_scatter(x=df_costs_leasing[df_costs_leasing.columns[0]], y=df_costs_leasing["Total"], mode="lines+markers",
                    name="Total", line=dict(color="orange",width=2), yaxis="y2")
    fig.update_layout(barmode="stack", title=f"{period}ly Cost Breakdown of {brand_2}-{model_2} for Leasing at {leasing_price} €/month", xaxis_title=period, legend=dict(orientation="h", y=1.15, x=0),
                      yaxis_title="Cost Components (€)", yaxis2=dict(title="Total Cost (€)", overlaying="y", side="right", range=[0, 50000], showgrid=False), width=800, height=500)
    st.plotly_chart(fig,use_container_width=True)


with c2:
    fig = go.Figure()
    for col in [ "Fuel","Insurance","Tax","Maintenance", "Down_Payment", "Zinsen", "Tilgung"]:
        fig.add_bar(x=df_costs_buying[period], y=df_costs_buying[col], name=col)
    fig.add_scatter(x=df_costs_buying[df_costs_buying.columns[0]], y=df_costs_buying["Total"], mode="lines+markers",
                    name="Total", line=dict(color="orange",width=2), yaxis="y2")
    fig.update_layout(barmode="stack", title=f"{period}ly Cost Breakdown of {brand_2}-{model_2} for Buying at {initial_buy} € initial buy", xaxis_title=period, legend=dict(orientation="h", y=1.15, x=0),
                      yaxis_title="Cost Components (€)", yaxis2=dict(title="Total Cost (€)", overlaying="y", side="right", range=[0, 50000], showgrid=False), width=800, height=500)
    st.plotly_chart(fig,use_container_width=True)

# st.dataframe(df_costs_leasing, use_container_width=True)
# st.dataframe(df_costs_buying, use_container_width=True)

# ------------------ Filtered Listings ------------------
st.header("Filtered Listings")
mask = (
    (df.brand == brand) &
    (df.model == model) &
    (df.age == age) &
    (df.distance <= distance)
)

if HAS_POWER:
    mask &= df.transmission_kw <= engine_power

st.dataframe(df[mask].reset_index(drop=True), use_container_width=True, hide_index=True,
             column_config={"image": st.column_config.ImageColumn("Image"), "price": st.column_config.NumberColumn("Price (€)", format="€ %.0f"),
                            "distance": st.column_config.NumberColumn("Distance (km)", format="%.0f km"),
                            "age": st.column_config.NumberColumn("Age (years)", format="%.0f"), "url": st.column_config.LinkColumn("Link", width="small")})


