# 🛰️ EcoEngine AI — Hackathon 3.1

EcoEngine AI is a high-performance geospatial analysis tool designed to assess ecological health and predict climate-driven stress. It uses real-time satellite data from Microsoft's Planetary Computer to generate a comprehensive **Ecological Stress Index (CESI)** and uses Machine Learning to simulate future environmental risk.

---

## 🚀 Key Features

- **🌍 Dynamic Location Search:** Never hardcode coordinates again. Type any city or region (e.g., "Munnar", "Amazon Rainforest", "Dehradun") and the AI will find it.
- **🌱 Advanced Carbon Pipeline:** Fixed and optimized carbon stock assessment using the **HGB (Harmonized Global Biomass)** and **Chloris Biomass** datasets.
- **🧠 Regional AI Training:** Automatically trains a **Random Forest** model on local environmental data to learn unique regional stressors.
- **🌡️ Climate Simulation:** Predictive AI modeling that simulates a **+2°C warming scenario** to visualize future ecological risk zones.
- **📊 Comprehensive Eco-Reporting:** Instant reports on Vegetation Health (NDVI), Moisture Stress (NDWI), Land Surface Temperature (LST), and Terrain Complexity.

---

## 🛠️ Installation & Setup

### 1. Requirements
Make sure you have Python 3.9+ installed.

### 2. Install Dependencies
Run these commands in your terminal one by one:
```bash
pip install pystac-client planetary-computer odc-stac xarray numpy matplotlib scikit-learn joblib requests pandas
```

*Note: `requests` and `pandas` are required for the new dynamic location and data handling features.*

---

## 📖 How to Use

1. **Run the Engine:**
   ```bash
   python main.py
   ```

2. **Enter Location:**
   When prompted, type the name of the area you want to analyze.
   > *Example: Enter a city or region: Nainital*

3. **View Results:**
   - **Terminal:** Check the console for the "ECOLOGICAL AI REPORT" and "TOP STRESS DRIVERS".
   - **Visuals:** Open the `sentinel_output` folder to see the generated `AI_PREDICTION_...png` maps.
   - **Models:** The regional AI model is saved as a `.joblib` file in the same folder.

---

## 📡 Data Sources
Powered by **Microsoft Planetary Computer**:
- **Sentinel-2 L2A:** Surface Reflectance & Indices
- **Landsat-C2-L2:** Thermal Infrared (LST)
- **HGB:** Harmonized Global Biomass (Carbon Stock)
- **Copernicus DEM:** Elevation & Slope
- **ESA WorldCover:** Land Classification

---

## ⚖️ License
This project is part of Hackathon 3.1. Feel free to fork and build upon it!
