# AI-Fault-Diagnosis
A MATLAB-based predictive maintenance system for robotic actuators. Uses Machine Learning and signal processing to extract features from vibration data and classify mechanical faults (like bearing wear), bridging mechanical engineering with AI diagnostics.

# AI-Based Fault Diagnosis for Robotic Actuators (Simulink & MATLAB)

![Project Banner](AI-Fault_Diagnosis/Actuator_model.png)

## 📋 Project Overview
This project focuses on the **predictive maintenance** of robotic actuators. Instead of relying on expensive physical prototypes, I developed a high-fidelity **Simulink model** to simulate the physical behavior of an actuator under various health conditions. 

The generated synthetic data was used to train and compare two Machine Learning models—**Support Vector Machine (SVM)** and **Convolutional Neural Network (CNN)**—to automatically classify mechanical faults.

## 🎯 Objectives
1.  **Simulation:** Create a physics-based model of a robotic actuator in Simulink.
2.  **Data Generation:** Simulate different failure modes to create a comprehensive dataset.
3.  **Diagnosis:** Implement and compare AI models to detect faults without manual signal filtering.

## 🛠️ Fault Classes
The system is designed to detect the following 5 conditions:
* ✅ **Healthy Condition** (Baseline)
* ⚠️ **Motor Fault** (e.g., winding short/eccentricity)
* ⚠️ **Gearbox Fault** (e.g., tooth wear)
* ⚠️ **Bearing Fault** (e.g., inner/outer race degradation)
* 🚨 **Mixed Fault** (Simultaneous occurrence of multiple failure modes)

## 🧠 Model Architecture & Methodology

### 1. Data Generation (Simulink)
* Modeled the electromechanical dynamics of the actuator (DC Motor + Gear Train).
* Faults were injected by modifying physical parameters (e.g., friction coefficients, inertia, resistance) within the Simulink blocks.
* **Data Strategy:** Raw sensor data was captured directly. **No manual filters** or signal pre-processing (like FFT) were applied, allowing the AI models to learn features directly from the raw time-series data.

### 2. Machine Learning Models
I implemented two distinct approaches to compare performance:

* **Approach A: Support Vector Machine (SVM)**
    * Used as a baseline classical ML model.
    * Classifies fault types based on hyperplanes in multidimensional space.

* **Approach B: Convolutional Neural Network (CNN)**
    * **Why CNN?** CNNs are powerful for time-series analysis as they can automatically extract hierarchical features from raw data, eliminating the need for manual feature engineering.
    * Architecture: [e.g., 1D-CNN layers -> Pooling -> Fully Connected Layer].

## 📊 Results & Comparison
The models were evaluated based on classification accuracy on a held-out test set.

| **CNN** | 
**Best Performance.** Successfully captured complex patterns in the raw data, particularly for the "Mixed Fault" class.   
| **SVM** |
**Good baseline** Struggled to distinguish between subtle fault signatures compared to the Deep Learning approach. 

