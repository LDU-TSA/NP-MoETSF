# NP-MoETSF

> A framework for high-dimensional multivariate time series forecasting

---

## 📖 Overview

This repository contains the official implementation of **NP-MoETSF**, a deep learning framework for multivariate time series forecasting.

The model is designed to handle:

* high-dimensional data
* complex cross-variable dependencies
* long-horizon forecasting scenarios

---

## 🧠 Method (Brief)

NP-MoETSF consists of two main components:

* **Structure Learning Module**
  Learns sparse dependencies between variables

* **Mixture-of-Experts Module**
  Dynamically models heterogeneous temporal patterns

> More details will be released after paper publication.

---

## 📂 Project Structure

```bash
NP-MoETSF/
├── data_provider/        
├── exp/                  
├── models/               
├── pytorch_wavelets/     
├── utils/                
├── scripts/              
├── run.py                
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/NP-MoETSF.git
cd NP-MoETSF
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python run.py
```

or

```bash
bash scripts/ETTh1.sh
```

---

## 📊 Datasets

Supported datasets include:

* ETT (ETTh1, ETTh2, ETTm1, ETTm2)
* Electricity
* Traffic
* Weather
* M4

---

## 📌 Notes

* The paper is currently under review.
* Detailed method description and full experimental results will be released later.

---

## 📜 License

MIT License
