# Fabric Thread Count Analyzer

## Introduction

Fabric quality assessment is crucial in the textile industry, with thread count being a key factor in determining durability, texture, and comfort. Traditionally, this process is manual and error-prone. This project automates the process using computer vision techniques, leveraging a Raspberry Pi Camera to capture fabric images and analyze warp (vertical) and weft (horizontal) threads per square inch.

## Key Features

- Faster and more accurate than manual counting
- Works with different fabric weave patterns
- Automated classification of dense and loose weaves
- Cost-effective solution for textile quality control

## Technologies Used

- **Programming Language**: Python
- **Image Processing**: OpenCV
- **Data Visualization**: Matplotlib
- **Numerical Computing**: NumPy, SciPy
- **Hardware**: Raspberry Pi
- **Camera**: Raspberry Pi Camera

## How It Works

1. **Capture Fabric Image**: The Raspberry Pi camera captures a high-resolution image of the fabric sample.
2. **Preprocess Image**: Converts to grayscale, removes noise, and enhances contrast using CLAHE.
3. **Apply Adaptive Thresholding**: Converts the fabric texture into a binary image for better thread visibility.
4. **Detect Threads Using Peak Detection**: Analyzes horizontal and vertical pixel intensity and detects warp and weft threads.
5. **Display and Save Results**: The detected thread count is displayed, and processed images are saved for verification.

## Hardware Setup and Installation

### 1. Install Raspberry Pi OS

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Raspberry Pi Imager
sudo apt install rpi-imager -y

# Open Raspberry Pi Imager
# Select Raspberry Pi OS (32-bit)
# Choose your microSD card (8GB minimum)
# Click Write and wait for installation to complete

# Enable SSH for remote access
touch /boot/ssh
```
##2. Connect to Raspberry Pi via SSH (PuTTY)
```bash
# Find the Raspberry Pi's IP Address
hostname -I
```
Open PuTTY
Enter Raspberry Pi's IP Address
Set Port: 22
Set Connection Type: SSH
Log in with:
Username: pi
Password: raspberry

## 3. Enable Camera Module
```bash
# Open Raspberry Pi Configuration Tool
sudo raspi-config

# Navigate to "Interfacing Options" > "Camera" and enable it
# Reboot Raspberry Pi to apply changes
sudo reboot
```
## 4. Install Required Libraries
```bash
pip install opencv-python numpy matplotlib scipy picamera
```

## Project Setup
1. Clone the Repository
``` bash
git clone https://github.com/yourusername/Fabric-Thread-Analyzer.git
cd Fabric-Thread-Analyzer
```
2. Run the Main Script
```bash
python main.py
```
## User Controls

- Press c: Capture an image and analyze fabric thread count
- Press q: Quit the program

## Expected Output

-Total Thread Count (Warp and Weft)
-Processed Image with Detected Threads
-Graphical Thread Projection Plots

## Future Enhancements
-AI-based Thread Classification - Train a neural network for more robust fabric analysis
-Mobile and IoT Integration - Develop a web/mobile app for remote monitoring


