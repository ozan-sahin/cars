# 🚗 Used Car Price Analysis & Total Cost of Ownership Dashboard

This project is an **interactive Streamlit dashboard** for analyzing used car prices and estimating **fair value, depreciation, resale value, and total cost of ownership (TCO)** using real-world data from German car marketplaces.

The app supports multiple datasets and provides tools for **price exploration, depreciation modeling, and financial decision-making** (buy vs lease).

---

## ✨ Features

### 📊 Exploratory Data Analysis

* Price distribution
* Top brands by listings
* Average price, median age, listing count
* Interactive charts (Plotly)

### 💰 Fair Price Estimator

* Estimate a fair market price based on:

  * Brand & model
  * Vehicle age
  * Mileage (distance)
  * Engine power (if available)
* Filters outliers automatically (95% quantiles)

### 📉 Depreciation Modeling

* Fits an **exponential depreciation curve**:

  [
  price(age) = a \cdot e^{-b \cdot age}
  ]

* Estimate:

  * Value at any future age
  * Resale value given purchase age & price
  * Reverse price → implied age

### 📈 Model Comparison

* Compare multiple models of the same brand
* Visualize price vs age curves

### 🚘 Total Cost of Ownership (TCO)

* Monthly or yearly cost breakdown:

  * Depreciation
  * Leasing (optional)
  * Insurance
  * Tax
  * Fuel
  * Maintenance
* Stacked bar chart with:

  * Component costs
  * Aggregated total cost on a **secondary axis**

### 🖼️ Rich Listings Table

* Vehicle images
* Formatted prices & mileage
* Clickable listing URLs
* Compact, readable layout

---

## 📁 Supported Datasets

* **Autoscout24**
* **Kleinanzeigen**

The app automatically adapts if certain columns (e.g. `transmission_kw`) are missing.

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **Pandas / NumPy**
* **Plotly**
* **SciPy** (curve fitting)

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/used-car-price-dashboard.git
cd used-car-price-dashboard
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

---

## 📌 Project Structure

```
├── app.py                     # Streamlit app
├── dataset2.csv               # Autoscout24 dataset
├── kleinanzeigen_cleaned.csv  # Kleinanzeigen dataset
├── requirements.txt
└── README.md
```

---

## 📐 Methodology Highlights

* Robust outlier removal using quantiles
* Relative depreciation for resale estimation
* Dataset-agnostic logic (column-aware)
* Compact, readable UI with consistent formatting

---

## 🚀 Future Improvements

* Confidence intervals for depreciation curves
* EV-specific modeling (battery degradation)
* Regional price adjustments
* Machine-learning-based fair price prediction
* Export reports (PDF)

---

## 📄 License

MIT License — feel free to use, modify, and extend.

---

## 👤 Author

**Ozan Sahin**
Head of Operations Continent @ Statkraft
M.Sc. Power Engineering (TUM)
B.Sc. Mechanical Engineering (ITU)

---

Just tell me 👍
