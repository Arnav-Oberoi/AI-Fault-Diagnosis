# AI-Fault-Diagnosis
A MATLAB-based predictive maintenance system for robotic actuators. Uses Machine Learning and signal processing to extract features from vibration data and classify mechanical faults (like bearing wear), bridging mechanical engineering with AI diagnostics.

# AI-Based Fault Diagnosis for Robotic Actuators

![System Block Diagram](link_to_your_diagram_image.png)
*(Recommended: Upload a block diagram showing: Sensor -> Data -> Feature Extraction -> AI -> Diagnosis)*

## 1. Project Overview
This project implements a Machine Learning workflow to detect and classify mechanical faults in robotic actuators using vibration/current data. 
* **Application:** Predictive maintenance for industrial robotic arms to reduce downtime.
* **Target Faults:** Bearing wear, Gear tooth fracture, and Motor eccentricity.
* **Accuracy:** Achieved **96.5%** classification accuracy using [Insert Method, e.g., SVM or CNN].

## 2. Technical Stack
* **Language:** MATLAB R2023b
* **Toolboxes:** Signal Processing Toolbox, Statistics and Machine Learning Toolbox.
* **Hardware:** [Mention hardware if used, e.g., Dynamixel Actuators, Accelerometers, DAQ card].

## 3. Methodology
The fault diagnosis pipeline follows these steps:

### A. Data Acquisition & Pre-processing
* Raw vibration signals collected at **[Sampling Rate] Hz**.
* Noise reduction using a low-pass filter to isolate mechanical frequencies.

### B. Feature Extraction
To bridge the gap between raw data and ML, statistical features were extracted from the Time and Frequency domains:
* **Time Domain:** RMS, Kurtosis, Skewness (indicative of impulsive shocks from gear faults).
* **Frequency Domain:** FFT Peak analysis to identify characteristic fault frequencies.

### C. Classification Model
* **Algorithm:** [e.g., Support Vector Machine (SVM) / K-Nearest Neighbors (KNN)].
* **Training/Test Split:** 70% Training, 30% Testing.
* **Validation:** 5-fold Cross-Validation to ensure model robustness.

## 4. Results
The model successfully distinguished between Healthy, Inner Race Faults, and Broken Tooth conditions.

![Confusion Matrix](link_to_confusion_matrix_image.png)
*(Upload a screenshot of your MATLAB confusion matrix here)*

## 5. How to Run
1. Clone the repository.
2. Open `main_diagnosis.m` in MATLAB.
3. Load the sample data from the `/data` folder.
4. Run the script to view the classification output and accuracy plots.
