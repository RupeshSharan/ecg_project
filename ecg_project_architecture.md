# ECG Project Architecture and Flowcharts

This document outlines the high-level architecture of the ECG Arrhythmia Intelligence system and the execution flow of the project's components.

## Overall System Architecture

The overall architecture demonstrates how raw data is transformed, used to train various uncompressed and compressed models, evaluated on edge-like conditions, and finally served via a Streamlit interface.

```mermaid
graph TD
    subgraph Data Layer
        A[Raw MIT-BIH WFDB Data] -->|Filtering & Segmentation| B(Processed Sequences)
        B -->|Train/Val/Test| C[(Data Splits - NPY)]
    end

    subgraph Model Zoo
        M1[1D-CNN]
        M2[ResNet1D]
        M3[BiLSTM]
        M4[Transformer1D]
    end

    subgraph Training & Compression Layer
        C --> T1[FP32 Baseline Training]
        C --> T2[Mixed Precision Training<br>FP16/BF16]
        T1 -.-> Q1[PTQ<br>Static/Dynamic INT8]
        T1 -.-> Q2[QAT<br>Quantization-Aware Training]
        
        T1 ==> M1 & M2 & M3 & M4
        T2 ==> M1 & M2 & M3 & M4
        Q1 ==> M1 & M2 & M3 & M4
        Q2 ==> M1 & M2 & M3 & M4
    end

    subgraph Evaluation & Interpretability
        E1[Metrics: Accuracy, F1, AUC]
        E2[Latency & Size Benchmarking<br>ONNX Export]
        E3[Interpretability:<br>Saliency, Grad-CAM, SHAP, LIME]
        
        M1 & M2 & M3 & M4 --> E1
        M1 & M2 & M3 & M4 --> E2
        M1 & M2 & M3 & M4 --> E3
    end

    subgraph User Interface Layer
        UI[Streamlit Dashboard]
        UI_I[Inference Tab]
        UI_M[Metrics Explorer]
        UI --> UI_I
        UI --> UI_M
        
        M1 & M2 & M3 & M4 -.->|Checkpoints| UI_I
        E1 & E2 -.->|JSON Metrics| UI_M
    end

    classDef dataset fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px;
    classDef model fill:#fff3e0,stroke:#ff9800,stroke-width:2px;
    classDef training fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;
    classDef ui fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px;

    class A,B,C dataset;
    class M1,M2,M3,M4 model;
    class T1,T2,Q1,Q2 training;
    class UI,UI_I,UI_M ui;
```

---

## 1. Core Pipeline Flowchart (`run_all.py`)

This flowchart illustrates the sequence of operations for model training, optimization, and evaluation.

```mermaid
flowchart TD
    Start([Start Pipeline]) --> P1(Phase 1: Preprocessing<br>Bandpass, Segment, Split)
    P1 --> P2(Phase 2: Train Baselines<br>FP32 Models)
    P2 --> P3(Phase 3: Mixed Precision<br>FP16 / BF16)
    P3 --> P4(Phase 4: Quantization<br>INT8 PTQ & QAT)
    P4 --> P5(Phase 5: Interpretability<br>Grad-CAM, Saliency, SHAP)
    P5 --> P6(Phase 6: Comparative Analysis<br>Metrics Aggregation)
    P6 --> P7(Phase 7: Ablation Studies<br>Granularity, Faithfulness)
    P7 --> End([End Pipeline])

    style Start fill:#2e7d32,stroke:#1b5e20,stroke-width:2px,color:#fff
    style End fill:#c62828,stroke:#b71c1c,stroke-width:2px,color:#fff
```

---

## 2. Streamlit Dashboard Inference Flow

This flowchart breaks down how a user's input gets processed and visualized in the live Streamlit Dashboard (`streamlit_dashboard.py`).

```mermaid
flowchart TD
    User([User Opens Dashboard]) --> Load[Load Checkpoints & Metrics]
    Load --> Select[User Selects Model Checkpoint]
    
    Select --> Source{Input Source?}
    
    Source -->|Test Split| Data1[Load Sample from X_test.npy]
    Source -->|Upload File| Data2[Parse .npy, .csv, or .txt]
    Source -->|Manual Values| Data3[Parse Comma-Separated Values]
    
    Data1 & Data2 & Data3 --> Coerce[Coerce Shape to 1D Array of size 360]
    
    Coerce --> Norm{Apply Z-score?}
    Norm -->|Yes| Z[Normalize Signal]
    Norm -->|No| Torch[Convert to PyTorch Tensor]
    Z --> Torch
    
    Torch --> Model[Model Forward Pass]
    Model --> Prob[Extract Softmax Probabilities]
    Model --> Sal[Compute 1D Saliency Map via Gradients]
    
    Prob --> UI1[Display Predicted Class & Confidence]
    Prob --> UI2[Render Altair Probability Bar Chart]
    Sal --> UI3[Plot Line Chart of Saliency overlay]
    
    UI1 & UI2 & UI3 --> Done([Wait for Next Input])

    style User fill:#1565c0,stroke:#0d47a1,stroke-width:2px,color:#fff
```
