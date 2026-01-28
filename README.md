# Energy-Efficient Fast Object Detection on Edge Devices for IoT Systems

Official implementation of **Energy-Efficient Fast Object Detection on Edge Devices for IoT Systems**, published in **IEEE Internet of Things Journal (2025)**.

This repository provides an efficient fast-moving object detection framework designed for real-time and low-power IoT edge devices.

---

## 🔍 Overview

Fast-moving object detection is challenging in IoT environments due to strict constraints on latency, power consumption, and hardware resources.  
This work proposes a lightweight detection framework that combines **frame difference–based motion detection** with **efficient AI classifiers**, avoiding computationally expensive end-to-end object detectors.

The proposed method achieves superior energy efficiency and lower inference latency while maintaining competitive accuracy, making it suitable for deployment on resource-constrained edge platforms.

---

## 🏗 Architecture

The proposed system consists of three main stages:

1. **Motion Detection (Frame Difference)**  
   Detects moving regions by computing pixel-wise differences between consecutive frames.

2. **Pre-processing**  
   Extracts regions of interest (ROIs), performs resizing and normalization, and prepares inputs for AI inference.

3. **AI Classification**  
   Classifies detected moving objects using lightweight deep learning models, including:
   - MobileNet
   - ResNet50
   - Inception-v4
   - Vision Transformer (ViT)

---

## 📊 Experimental Results

The proposed approach is evaluated on multiple edge platforms:
- AMD Alveo™ U50 FPGA
- NVIDIA Jetson Orin Nano
- Hailo-8™ AI Accelerator

Experimental results demonstrate:
- Up to **28.31% accuracy improvement**
- Up to **39.3% reduction in inference latency**
- Up to **3.6× efficiency improvement** compared to end-to-end object detection methods

---

## 📄 Paper

**Energy-Efficient Fast Object Detection on Edge Devices for IoT Systems**  
*IEEE Internet of Things Journal*, vol. 12, no. 11, pp. 16681–16693, 2025.

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@ARTICLE{10879008,
  author={Nurul Achmadiah, Mas and Ahamad, Afaroj and Sun, Chi-Chia and Kuo, Wen-Kai},
  journal={IEEE Internet of Things Journal}, 
  title={Energy-Efficient Fast Object Detection on Edge Devices for IoT Systems}, 
  year={2025},
  volume={12},
  number={11},
  pages={16681-16694},
  keywords={Object detection;Internet of Things;Accuracy;Real-time systems;Image edge detection;Object recognition;Artificial intelligence;Hardware acceleration;Computational modeling;Cameras;AI classifier;energy efficiency;fast-moving object detection (FMOD);high-latency;real-time performance},
  doi={10.1109/JIOT.2025.3536526}}
