# Source Localization with Kalman Filter and MUSIC Algorithm  

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## English US

### Overview
This repository implements a source localization system using:
- **MUSIC Algorithm** for Direction of Arrival (DoA) estimation
- **Kalman Filter** (KF) and **Extended Kalman Filter** (EKF) for position tracking
- Geometric intersection methods for multi-array position estimation

### Key Features
- 🎯 2D position estimation using circular microphone arrays
- 📡 MUSIC algorithm implementation for angle estimation
- 🧭 Kalman Filter for sensor fusion and trajectory smoothing
- 📈 Multiple array support for improved accuracy
- 🏢 Polygon constraint handling for indoor scenarios

### Directory Structure

- **EKF/** (Extended Kalman Filter)
  - `ekf_utils.py`: EKF mathematical operations
  - `main.py`: Extended Kalman Filter demo runner

- **KF/** (Standard Kalman Filter)
  - `kalman_filter.py`: 2D Kalman Filter class
  - `position_estimation.py`: Geometric position estimation functions
  - `main.py`: Basic Kalman Filter demo runner

  - Utility Tools and Line Intersection Method
  - `array_utils.py`: Circular array generation functions
  - `geometry_utils.py`: Polygon operations (sampling, location check)
  - `music_utils.py`: MUSIC algorithm implementation
  - `main.py`: MUSIC algorithm demo with line intersection

## Türkçe 🇹🇷
# Kalman Filtre ve MUSIC Algoritması ile Kaynak Konumlandırma  
### Genel Bakış
Bu depo şunları içeren bir kaynak konumlandırma sistemi uygular:

- **MUSIC Algoritması** ile Geliş Açısı (DoA) kestirimi
- Konum takibi için **Kalman Filtresi** (KF) ve **Genişletilmiş Kalman Filtresi** (EKF)
- Çoklu dizi konum kestirimi için geometrik kesişim yöntemleri

### Temel Özellikler
- 🎯 Dairesel mikrofon dizileri ile 2B konum kestirimi
- 📡 Açı kestirimi için MUSIC algoritması uygulaması
- 🧭 Sensör füzyonu ve yörünge düzeltme için Kalman Filtresi
- 📈 Hassasiyet artırımı için çoklu dizi desteği
- 🏢 Kapalı alan senaryoları için poligon kısıtlaması

### 📂 Dizin Yapısı

- **EKF/** (Genişletilmiş Kalman Filtre)
  - `ekf_utils.py`: EKF matematiksel operasyonları
  - `main.py`: Genişletilmiş Kalman Filtre demo çalıştırıcısı

- **KF/** (Standart Kalman Filtresi)
  - `kalman_filter.py`: 2D Kalman Filtre sınıfı
  - `position_estimation.py`: Geometrik konum kestirim fonksiyonları
  - `main.py`: Temel Kalman Filtre demo çalıştırıcısı

- Yardımcı Araçlar ve Çizgi Kesişimi Metodu
  - `array_utils.py`: Dairesel dizi üretim fonksiyonları
  - `geometry_utils.py`: Poligon işlemleri (örnekleme, konum kontrol)
  - `music_utils.py`: MUSIC algoritması implementasyonu
  - `main.py`: Çizgi kesişimi ile MUSIC algoritması demo çalıştırıcısı
