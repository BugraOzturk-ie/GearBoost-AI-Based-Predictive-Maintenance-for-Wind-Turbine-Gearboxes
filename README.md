# GearBoost: AI-Based Predictive Maintenance for Wind Turbine Gearboxes

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

**GearBoost** is a comprehensive machine learning project focused on predictive maintenance for wind turbine gearboxes using ensemble boosting algorithms. The project compares three state-of-the-art gradient boosting methods—**XGBoost**, **LightGBM**, and **CatBoost**—to predict gearbox failures using real-world industrial sensor data from wind turbines.

This project was developed as part of the **TÜBİTAK 2209/B University Students Research Projects Support Program** in collaboration with **Enerjisa Üretim A.Ş.**, one of Turkey's leading energy production companies.

### Key Features

- **Real Industrial Data**: Analysis based on actual sensor data from operational wind turbines
- **Time-Series Analysis**: Proper handling of temporal dependencies using TimeSeriesSplit cross-validation
- **Advanced Feature Engineering**: Automated creation of lag features, rolling statistics, and temporal patterns
- **Model Comparison**: Comprehensive evaluation of XGBoost, LightGBM, and CatBoost algorithms
- **Industry-Validated**: Methodology informed by real-world predictive maintenance practices

## Problem Statement

Wind turbine gearboxes are critical components that experience significant stress and are prone to failure. Unplanned downtime due to gearbox failures can result in:

- High maintenance costs
- Production losses
- Extended repair periods
- Safety risks

This project aims to predict gearbox failures before they occur, enabling proactive maintenance scheduling and reducing overall operational costs.

## Dataset

The dataset consists of time-series sensor data collected at **10-minute intervals** from wind turbines, including:

### Sensor Variables
- **Power Output**: `BRS.T35_Power`
- **Oil Temperatures**: `BRS.T35_T_GBX_OIL_1`, `BRS.T35_T_GBX_OIL_2`
- **Bearing Temperatures**: 
  - High-Speed Shaft: `BRS.T35_T_GBX_T1_HSS`, `BRS.T35_T_GBX_T3_HSS`
  - Intermediate Shaft: `BRS.T35_T_GBX_T1_IMS`, `BRS.T35_T_GBX_T3_IMS`
- **Gear Bearing Temperature**: `BRS.T35_T_GEAR_BEAR`
- **Environmental Data**: `BRS.T35_Temp_Ambient`, `BRS.T35_WIND_DIR`, `BRS.T35_Wind_speed`

### Dataset Statistics
- **Total Records**: 57,024 observations
- **Time Span**: Approximately 4 months of continuous monitoring
- **Source**: Anonymized data from Enerjisa wind energy facilities

## Repository Structure

```
GearBoost/
│
├── Gearboost_data_preparation.ipynb    # Data preprocessing and feature engineering
├── Gearboost_CV_step_by_step.ipynb     # Model training and cross-validation
├── README.md                            # Project documentation
└── requirements.txt                     # Python dependencies
```

## Methodology

### 1. Data Preparation (`Gearboost_data_preparation.ipynb`)

#### Preprocessing Steps:
1. **Time Alignment**: Converting timestamps to datetime index with 10-minute intervals
2. **Data Resampling**: Regularizing time series using median aggregation
3. **Missing Value Handling**: Time-based interpolation for gaps up to 1 hour
4. **Data Cleaning**: Removal of extended missing data periods

#### Feature Engineering:
- **Lag Features**: Creating historical values at lags [1, 3, 5] time steps
- **Rolling Statistics**: 
  - Moving averages (windows: 3, 5, 10 time steps)
  - Rolling standard deviations
- **Temporal Derivatives**:
  - First-order differences
  - Percentage changes
- **Anomaly Detection**: Z-score based anomaly labeling (threshold = 3)

### 2. Model Development (`Gearboost_CV_step_by_step.ipynb`)

#### Algorithms Compared:
1. **XGBoost** (eXtreme Gradient Boosting)
2. **LightGBM** (Light Gradient Boosting Machine)
3. **CatBoost** (Categorical Boosting)

#### Training Strategy:
- **Cross-Validation**: TimeSeriesSplit with 5 folds
- **Early Stopping**: Prevents overfitting with 50-round patience
- **Evaluation Metrics**:
  - ROC AUC Score
  - Accuracy
  - F1 Score
  - Precision
  - Recall

#### Model Pipeline:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Standard pipeline for all models
pipeline = make_pipeline(
    StandardScaler(),
    Model()
)
```

#### Threshold Optimization:
- F1-score optimization to find the best classification threshold
- Out-of-fold (OOF) predictions for unbiased performance estimation

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GearBoost.git
cd GearBoost
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Required Packages
```
numpy
pandas
matplotlib
scikit-learn
xgboost
lightgbm
catboost
openpyxl
```

## Usage

### 1. Data Preparation
Run the data preparation notebook to preprocess raw sensor data and engineer features:

```bash
jupyter notebook Gearboost_data_preparation.ipynb
```

**Input**: Raw Excel file with sensor readings  
**Output**: `GearBoost Prepared Data.xlsx` with engineered features

### 2. Model Training and Evaluation
Execute the cross-validation notebook to train and compare models:

```bash
jupyter notebook Gearboost_CV_step_by_step.ipynb
```

**Expected Outputs**:
- Cross-validation performance metrics for each model
- ROC curves comparison
- Optimal threshold values
- Feature importance rankings

## Results

The project provides a comprehensive comparison of three leading boosting algorithms specifically for wind turbine gearbox failure prediction. Key findings include:

- **Performance Metrics**: Detailed ROC AUC, F1, precision, and recall scores for each algorithm
- **Model Comparison**: Side-by-side evaluation under identical time-series cross-validation conditions
- **Industry Validation**: Results validated against real-world deployment of CatBoost in production environments

*Note: Specific numerical results are documented in the project notebooks.*

## Key Insights

1. **Time-Series Considerations**: Proper temporal validation is critical for realistic performance estimation
2. **Feature Engineering Impact**: Lag features and rolling statistics significantly improve predictive power
3. **Early Stopping**: Essential for preventing overfitting in boosting algorithms
4. **Threshold Tuning**: F1-optimized thresholds provide better balance than default 0.5 cutoff

## Future Work

Potential extensions of this project:

- [ ] Real-time prediction system deployment
- [ ] Integration with SCADA systems
- [ ] Multi-component failure prediction (generator, blades, etc.)
- [ ] Deep learning approaches (LSTM, Transformer models)
- [ ] Explainability analysis (SHAP values, feature importance)
- [ ] Cost-benefit analysis of predictive maintenance strategies

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:

- Bug fixes
- Documentation improvements
- New feature engineering techniques
- Alternative modeling approaches
- Performance optimizations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **TÜBİTAK** for funding through the 2209/B program
- **Enerjisa Üretim A.Ş.** for providing industrial sensor data
- **Manisa Celal Bayar University** for academic support
- **Advisor**: Öğr. Gör. Orkun Teke

## Contact

**Project Lead**: Buğra Öztürk

For questions or collaboration opportunities, please open an issue in this repository.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{gearboost2024,
  author = {Öztürk, Buğra},
  title = {GearBoost: AI-Based Predictive Maintenance for Wind Turbine Gearboxes},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/GearBoost}}
}
```

---

**Project Period**: March 2024 - September 2025  
**Program**: TÜBİTAK 2209/B - University Students Research Projects Support Program
