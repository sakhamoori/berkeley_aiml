# README: Top Feature Insights for Used Car Price Prediction  
**Audience:** Used Car Dealers | **Goal:** Inventory Optimization & Pricing Strategy  

### **Overview**  
This analysis identifies the **top 20 features** influencing used car prices, derived from a predictive modeling exercise. The insights are prioritized to help dealers align inventory with market demand and maximize profitability.  

1. Started off reading the csv followed with cleaning up Nan and droping any rows with price =0. 
2. Created a histplot to obsever the price trend of the used cars
3. Dropped off columns that aren't required
4. Used SimpleImputer to create Strategy for year and odometer readings
5. Transforming Price column & converting int to float
6. Training the model with a Preprocessor to transform the dataset
7. Created Model to perform all the regression models
8. Performing evaulation using RamdonForestRegression
9. Plotting the Barcart

### **Key Features & Interpretations**  
#### **1. Most Impactful Features**  
- **`cat_transmission_other`**  
  Non-standard transmissions (e.g., automated manual, CVT) show strong pricing influence. These may appeal to niche buyers.  
- **`cat_state_f1`**  
  Geographic location (e.g., state/region "F1") significantly affects prices. Tailor inventory to local preferences (e.g., trucks in rural areas).  
- **`num_year`**  
  Newer vehicles (<5 years old) drive higher prices due to lower depreciation and modern features.  
- **`num_odometer`**  
  Lower mileage (<75,000 miles) strongly correlates with higher valuations.  
- **High-Performance Engines** (`cat_cylinders_10 cylinders`, `8 cylinders`)  
  Larger engines are valued in trucks/SUVs, reflecting demand for power and utility.  

#### **2. High-Value Models**  
- **Top Models:** Toyota Tundra (4WD), Dodge Durango, BMW X6 M, Audi A7, Lexus TKX.  
- **Action:** Prioritize these models for faster turnover and premium pricing.  

#### **3. Fuel & Vehicle Type**  
- **Diesel Vehicles** (`cat_fuel_diesel`)  
  Diesel engines hold higher value, likely due to durability and fuel efficiency.  
- **Trucks/SUVs** (`cat_type_truck`, `cat_type_SUV`)  
  Aligns with broader market trends favoring utility vehicles.  

#### **4. Aesthetic & Condition**  
- **Paint Color** (`cat_paint_color_black`)  
  Black cars may command slight price premiums due to aesthetic appeal.  
- **Condition** (`cat_condition_fair`)  
  Surprisingly low importance—buyers prioritize age/mileage over minor wear.  

---

### **Recommendations for Dealers**  
1. **Inventory Prioritization**  
   - Source newer models (2018+) with mileage under 75,000.  
   - Stock trucks/SUVs (e.g., Tundra, Durango) and luxury models (BMW X6 M, Audi A7).  
2. **Regional Adjustments**  
   - Analyze demand in "State F1" (or similar regions) for location-specific trends.  
3. **Market Niche Opportunities**  
   - Highlight vehicles with rare transmissions (`cat_transmission_other`) or large engines (8/10 cylinders).  
4. **Pricing Strategy**  
   - Test premium pricing for black-colored cars and diesel-powered vehicles.  
