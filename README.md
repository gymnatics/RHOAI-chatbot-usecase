## 🔍 Job Role Prediction & Recommendation System

This project focuses on building an intelligent job recommendation engine that predicts the most suitable **job role** for a candidate based on structured data (e.g., experience, qualifications, location, etc.). After identifying the best-fit role, the system goes one step further by recommending relevant **job titles** commonly associated with that role — enhancing the user’s job search experience.

### ✨ Key Features
- **Exploratory Data Analysis (EDA):** Includes outlier detection, skewness analysis, and visualization of categorical feature distributions using CDFs.
- **Data Cleaning:** Handles missing values, duplicate entries, and categorical simplification (e.g., grouping rare categories as "Others").
- **Statistical Insights:** Uses **Cramér’s V** to detect high correlations between categorical features (e.g., `Job Title` ↔ `Role`, `Location` ↔ `Country`) to guide feature selection.
- **Model Training:** Trains a machine learning model to predict the best-fit `Role` for a candidate using features like experience, salary expectations, company size, etc.
- **Post-Prediction Mapping:** Maps predicted roles to a list of realistic job titles based on existing data, even though `Job Title` is excluded during training to avoid leakage.

### 📦 Potential Use Cases
- Personalized job matching platforms  
- Career advisory tools  
- HR candidate screening systems
