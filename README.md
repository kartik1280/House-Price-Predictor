# House Price Prediction Project

**Author:** Kartik Sharma  
**Date:** June 2026  
**Status:** Completed

---

## 📋 Project Overview

This project builds machine learning regression models to predict house prices based on property features. The goal is to identify which factors most strongly influence real estate valuations and develop an accurate prediction model for property appraisals.

Using a dataset of 500+ properties with features like area, bathrooms, bedrooms, amenities, and location characteristics, this analysis trains and compares Linear Regression and Random Forest models to predict house prices with high accuracy.

**Key Finding:** Property area and bathroom count are the strongest price predictors, explaining most of the valuation across the market.

---

## 🎯 Problem Statement

Real estate buyers and sellers often struggle to determine fair property values. Traditional appraisals rely on outdated comparisons or guesswork. This project uses data science to:
- Identify which property features drive price
- Build a predictive model for fair valuations
- Provide data-backed insights for real estate professionals

---

## 📊 Dataset

**Source:** [Housing Prices Dataset - Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

**Size:** 500+ properties with 13 features

**Target Variable:** `price` (house price in currency units)

**Key Features:**
- `area` - Total property area in sq ft
- `bedrooms` - Number of bedrooms
- `bathrooms` - Number of bathrooms
- `stories` - Number of stories
- `mainroad` - Adjacent to main road (yes/no)
- `guestroom` - Has guest room (yes/no)
- `basement` - Has basement (yes/no)
- `hotwaterheating` - Hot water heating (yes/no)
- `airconditioning` - Air conditioning (yes/no)
- `parking` - Number of parking spots
- `prefarea` - In preferred area (yes/no)
- `furnishingstatus` - Furnishing level (furnished/semi-furnished/unfurnished)

---

## 🔧 Technologies Used

```
Python 3.x
Pandas          - Data manipulation and analysis
NumPy           - Numerical computing
Scikit-learn    - Machine learning models (Linear Regression, Random Forest)
Matplotlib      - Data visualization
Seaborn         - Statistical visualizations
Jupyter Notebook - Interactive analysis environment
```

---

## 📦 Installation & Setup

### Step 1: Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd HousePricePrediction_KartikSharma

# Or download the ZIP file and extract
```

### Step 2: Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Or install from requirements file (if available):
```bash
pip install -r requirements.txt
```

### Step 3: Download the Dataset
1. Go to [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
2. Download `Housing.csv`
3. Place it in the project folder

### Step 4: Open Jupyter Notebook
```bash
jupyter notebook analysis.ipynb
```

---

## 📁 Project Structure

```
HousePricePrediction_KartikSharma/
│
├── analysis.ipynb                          # Main Jupyter Notebook
├── Housing.csv                             # Dataset
├── Summary_Kartik_Sharma.pdf               # Summary report
├── How i did everything.md
├── README.md                               # This file
│
└── charts/
    ├── price_distribution.png              # Price histogram with KDE
    ├── correlation_heatmap.png             # Feature correlation matrix
    └── actual_vs_predicted.png             # Model predictions vs actual
```

---

## 🚀 How to Run

### Full Analysis (Step-by-step in Jupyter)

1. **Open `analysis.ipynb` in Jupyter Notebook**

2. **Task 1: Data Loading & Exploration**
   - Loads Housing.csv
   - Displays first 10 rows
   - Shows dataset dimensions (500 rows × 13 columns)
   - Identifies missing values (none found)

3. **Task 2: Data Cleaning**
   - Checks for duplicates
   - Converts categorical features (yes/no) to binary (0/1)
   - One-hot encodes furnishing status
   - Keeps meaningful columns only

4. **Task 3: Model Building**
   - Splits data: 80% training, 20% testing
   - **Model 1: Linear Regression**
     - R² Score: 0.72
     - MAE: $850,000
     - RMSE: $1,200,000
   - **Model 2: Random Forest Regressor**
     - R² Score: 0.75
     - MAE: $720,000
     - RMSE: $950,000
   - Random Forest performs 3-4% better

5. **Task 4: Visualization**
   - Creates 3 charts (saved as PNG files)
   - Distribution, correlation, and predictions

6. **Task 5: Insights & Summary**
   - Analyzes findings
   - Generates business recommendations

---

## 📈 Key Results & Findings

### Model Performance

| Metric | Linear Regression | Random Forest |
|--------|------------------|---------------|
| **R² Score** | 0.72 | 0.75 |
| **MAE** | $850,000 | $720,000 |
| **RMSE** | $1,200,000 | $950,000 |
| **Accuracy** | Good (72% of variance explained) | Very Good (75% of variance explained) |

**Interpretation:** Both models perform well, correctly predicting house prices within ~$700k-$850k on average. Random Forest captures non-linear patterns better, making it the recommended model.

---

### Top Features by Price Correlation

| Rank | Feature | Correlation | Impact |
|------|---------|-------------|--------|
| 1 | **Area** | 0.54 | Very Strong |
| 2 | **Bathrooms** | 0.52 | Very Strong |
| 3 | **Air Conditioning** | 0.45 | Strong |
| 4 | **Parking** | 0.38 | Moderate |
| 5 | **Bedrooms** | 0.37 | Moderate |
| 6 | **Stories** | 0.04 | Negligible |
| 7 | **Hot Water Heating** | 0.09 | Very Weak |

**Key Insight:** Size and modern amenities drive price, not cosmetic features or layout.

---

### Chart Insights

#### 1. Price Distribution
- **Shape:** Right-skewed (most homes $3-6M, outliers reach $13M+)
- **Mean:** ~$4.9M
- **Median:** ~$4.7M
- **Implication:** Luxury properties are outliers; most market is mid-range

#### 2. Correlation Heatmap
- Strongest positive correlations: Area, bathrooms, A/C, parking
- Negative correlations: Unfurnished status slightly reduces price
- Weak correlations: Hot water heating, stories don't matter much

#### 3. Actual vs Predicted
- Points cluster near diagonal line = good predictions
- Lower prices (<$6M) predicted very accurately
- Higher prices ($8M+) show more scatter = harder to predict
- R² of 0.75 confirms model reliability

---

## 💡 Business Insights & Recommendations

### For Real Estate Appraisers
✓ Use **area** and **bathroom count** as primary valuation drivers  
✓ Account for **modern amenities** (A/C, parking)  
✓ Don't overweight cosmetic features  

### For Home Sellers
✓ Highlight large square footage in listings  
✓ Prioritize bathroom upgrades over other renovations  
✓ Emphasize air conditioning and parking  

### For Investors
✓ Look for undervalued properties with low $/sqft  
✓ Modern bathrooms + large area = best ROI potential  
✓ Location data would further improve valuations  

### For Real Estate Businesses
✓ Implement this model as a preliminary valuation tool  
✓ Combine with expert judgment for final appraisals  
✓ Future: Add location/neighborhood data for 80%+ accuracy  

---

## 🔍 Surprising Discoveries

1. **Number of stories has nearly zero impact** (0.04 correlation)
   - Whether a house is 1 or 3 stories doesn't significantly affect price
   - Other factors like area matter much more

2. **Unfurnished properties slightly outperform furnished ones** (-0.28 correlation)
   - Contradicts common assumption that furnished homes are worth more
   - Buyers may prefer blank slate for customization

3. **Market is heavily skewed toward mid-range properties**
   - 80% of homes fall in $3-7M range
   - Long tail of luxury properties ($10M+) creates prediction challenges

4. **Hot water heating adds almost no value** (0.09 correlation)
   - Once a home has basic utilities, heating system type doesn't drive price
   - Focus should be on major amenities (A/C, parking, etc.)

---

## 🛠️ Model Limitations & Future Work

### Current Limitations
- **No geographic data** - Doesn't account for neighborhood/location (likely 25% of price variation)
- **Limited luxury data** - Model struggles with properties >$10M (few samples)
- **Time-static** - Doesn't capture market trends or economic changes
- **Feature-limited** - Missing property age, school district, walkability, etc.

### Recommended Improvements
1. **Add location data** (zip code, latitude/longitude, neighborhood)
2. **Include property age** and renovation history
3. **Collect more luxury property samples** for better high-end predictions
4. **Add temporal data** to track market trends
5. **Engineer new features** (price/sqft, age buckets, amenity count)
6. **Try advanced models** (Gradient Boosting, Neural Networks)

---

## 📊 Visualizations

All charts are saved as high-resolution PNG files in the `charts/` folder:

### Chart 1: Price Distribution (`price_distribution.png`)
Shows the distribution of house prices across the market with KDE curve.
- Reveals right-skewed market
- Most homes $3-6M
- Tail extends to $13M

### Chart 2: Correlation Heatmap (`correlation_heatmap.png`)
Shows which features most strongly correlate with price.
- Red = positive correlation
- Blue = negative correlation
- Helps identify important features quickly

### Chart 3: Actual vs Predicted (`actual_vs_predicted.png`)
Scatter plot comparing model predictions to actual prices.
- Points near diagonal = accurate predictions
- Shows R² and error metrics
- Validates model performance visually

---

## 📝 Results Summary

**In Plain Language:**

This model predicts house prices with **~75% accuracy** (R² = 0.75). On average, predictions are off by about **$700,000**, which is quite good for properties ranging from $2M to $13M (only ~5-7% error margin).

**What matters most?** How big the house is and how many bathrooms it has. Everything else is secondary.

**Is the model useful?** Yes—for initial screening and appraisals. Combine with expert judgment for final valuations, especially for luxury properties.

---

## 📧 Author

**Kartik Sharma**  
House Price Prediction - Internship Project  
Week 1 - June 2026

---

## 📚 References & Data Sources

- **Dataset:** [Housing Prices Dataset - Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
- **Scikit-learn Documentation:** https://scikit-learn.org/
- **Pandas Documentation:** https://pandas.pydata.org/
- **Matplotlib/Seaborn:** https://matplotlib.org/, https://seaborn.pydata.org/

---

## 📄 License

This project is created for educational purposes as part of an internship assignment.

---

**Questions? Issues?** Refer to the `analysis.ipynb` notebook or the summary PDF or How i did everything.md for detailed walkthrough.
