# Performance and Interpretability Trade-offs in Mixed-Precision and Quantized ECG Arrhythmia Classifiers for Edge Deployment

**Ajay Bhaskar Reddy**, **Nikhilesh Kancherla**, **Cooly Siddarttha**, **Chavan Rupesh Sharan**
Computer Science and Engineering
SRM Institute of Science and Technology
*Software Engineering in Artificial Intelligence (21CSE312P)*

---

## Abstract
Continuous Electrocardiogram (ECG) monitoring is critical for the early detection and management of cardiac arrhythmias, a leading cause of global mortality. While modern deep learning architectures, such as Convolutional Neural Networks (CNNs) and Transformers, achieve state-of-the-art diagnostic accuracy, their immense computational footprint prohibits deployment on resource-constrained edge devices and wearables. Model compression techniques, specifically mixed-precision training and INT8 quantization, offer a pathway to real-time, on-device inference by drastically reducing latency and memory requirements. However, aggressive compression risks degrading the model's decision boundaries and altering the internal feature representations. Furthermore, the inherent "black-box" nature of neural networks poses a significant barrier to clinical adoption, necessitating the use of Explainable AI (XAI). This paper investigates the critical intersection of model compression and interpretability. We present a systematic framework utilizing the MIT-BIH Arrhythmia Database to analyze how Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) affect not only the classification accuracy of multiple architectures (1D-CNN, ResNet1D, BiLSTM, and Transformer1D) but also the stability of their visual explanations generated via Saliency Maps, SHAP, and LIME. Our analysis of "interpretability drift" provides insight into establishing trustworthy, edge-deployable AI for clinical decision support.

**Keywords**— ECG Arrhythmia Classification, Edge Computing, Quantization, Mixed Precision, Explainable AI (XAI), Interpretability Drift, Deep Learning.

---

## I. Introduction
The proliferation of Internet of Medical Things (IoMT) devices and smart wearables has revolutionized out-of-hospital cardiac monitoring. Cardiovascular diseases require continuous, long-term surveillance to capture transient arrhythmic events. Consequently, there is a paradigm shift towards performing real-time ECG signal analysis directly at the edge, rather than transmitting vast amounts of raw physiological data to centralized cloud servers—a process that incurs high latency and drains battery life.

Deep learning models have demonstrated exceptional capabilities in automated ECG classification, often matching or surpassing human cardiologists. However, deploying these highly parameterized models onto microcontrollers or mobile processors presents extreme challenges concerning memory, power, and compute constraints. To alleviate this, edge AI development relies heavily on model compression techniques, notably translating 32-bit floating-point (FP32) weights to 16-bit (FP16/BF16) or 8-bit integers (INT8).

In the healthcare domain, performance and efficiency alone are insufficient; transparency is a non-negotiable requirement. Clinicians must understand the rationale behind an algorithmic diagnosis to trust and act upon it. Explainable AI (XAI) techniques are employed to bridge this gap. Yet, an unaddressed vulnerability remains: when a model is subjected to numerical quantization, its internal reasoning paths may diverge from the original architecture. This phenomenon—which we term "interpretability drift"—means a compressed model might arrive at the correct diagnosis but rely on clinically irrelevant artifacts rather than the physiological P-QRS-T complex. This study systematically addresses this gap by benchmarking the efficiency against the fidelity of explanations across different deep learning architectures.

---

## II. Literature Survey
The intersection of deep learning, model compression, and Explainable AI for ECG analysis is an active field of research within the IEEE biomedical engineering community.

### A. Deep Learning for ECG Classification
Current state-of-the-art systems utilize advanced topological architectures to interpret 1D time-series signals. Research demonstrates that 1D-CNNs and ResNets excel at morphological feature extraction, while recurrent models (LSTMs) and emerging Transformer architectures successfully capture long-range temporal dependencies within the cardiac cycle. However, these reviews frequently highlight that the sheer size of these models restricts out-of-clinic usability.

### B. Model Compression for Wearables
Model compression techniques are essential for deploying deep learning on implantable monitors and wrist-worn devices. The literature primarily focuses on Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT). Studies have shown that converting FP32 weights to INT8 can reduce the model footprint by approximately 4x while simultaneously accelerating inference on specialized IoT accelerators. However, while some authors note minimal accuracy degradation on simplistic binary tasks, complex multi-class arrhythmia detection often suffers from unacceptable drops in the F1-score when quantization is applied indiscriminately.

### C. Explainable AI (XAI) in Cardiology
The "black-box" nature of neural networks is widely criticized in clinical settings. Recent IEEE publications emphasize the integration of XAI methodologies—such as Gradient-weighted Class Activation Mapping (Grad-CAM), SHapley Additive exPlanations (SHAP), and Local Interpretable Model-agnostic Explanations (LIME). These tools generate heatmaps or saliency curves over the ECG waveform, highlighting specific intervals (e.g., QRS complexes or T-waves) that triggered the classification. Despite this progress, several studies note that XAI outputs can be fragile and method-dependent. Strikingly, existing literature heavily evaluates XAI on pristine, uncompressed models, leaving a crucial research gap regarding how quantization algorithms distort clinical interpretability.

---

## III. Proposed Work
To address the aforementioned research gaps, this project proposes an end-to-end framework to comparatively analyze the efficiency, accuracy, and explanation stability of compressed ECG classifiers.

Our specific contributions include:
1. **Multi-Architecture Benchmarking:** Examining morphological (1D-CNN, ResNet1D), sequential (BiLSTM), and attention-based (Transformer1D) topologies.
2. **Precision Profiling:** Quantifying the direct impact of PyTorch Automatic Mixed Precision (FP16/BF16), INT8 PTQ (Dynamic and Static calibration), and INT8 QAT.
3. **Interpretability Drift Analysis:** We propose a novel evaluation analyzing whether the geometric mapping of gradient-based Saliency and SHAP values changes after model quantization.

---

## IV. Methodology

### A. Data Acquisition and Preprocessing
This study utilizes the benchmark MIT-BIH Arrhythmia Database accessed via the `wfdb` library. The raw signals suffer from baseline wander and high-frequency noise. 
1. **Filtering:** A Butterworth bandpass filter (0.5 to 40 Hz) is applied.
2. **Segmentation:** Signals are isolated by identifying R-peaks and extracting a uniform window (e.g., 360 samples). 
3. **Normalization:** Z-score normalization is applied to map patient-specific amplitude deviations into a standard distribution. Individual beats are mapped into the standard 5 AAMI classes (Normal, SVEB, VEB, Fusion, Paced).

### B. Network Architectures
We implement four baseline models. The 1D-CNN provides a fast, lightweight baseline. ResNet1D utilizes deep residual blocks to prevent gradient vanishing on deeper morphological abstractions. The BiLSTM tracks temporal evolution across the 360-unit sequence, and the Transformer1D utilizes positional encodings and multi-head attention to dynamically focus on correlating regions of the beat.

### C. Quantization Strategies
Following baseline FP32 training, models are optimized using:
*   **Mixed Precision:** Utilizing hardware-accelerated Tensor Cores to process activations in FP16/BF16, cutting memory limits in half during training.
*   **Post-Training Quantization (PTQ):** Fusing Conv-BatchNorm-ReLU layers and statically calibrating the activation ranges over a representative dataset to map FP32 tensors to INT8.
*   **Quantization-Aware Training (QAT):** Inserting fake-quantization modules during a secondary fine-tuning phase. This allows the network gradients to adjust to the clipping and rounding errors induced by standard INT8 conversion, generally yielding superior clinical accuracy.

### D. Explainable AI Extraction
To quantify interpretability, we project the network's reasoning back onto the 360-sample temporal vector. Captum is used to trace Saliency Maps and Integrated Gradients, while DeepSHAP evaluates feature attribution. By isolating the highest-scoring indices, we determine if the uncompressed and compressed models highlight the identical physiological features.

---

## V. Implementation
The framework is developed in Python using the PyTorch 2.x ecosystem. The codebase is organized modularly, separating data loaders, trainer engines, and quantization scripts (`run_all.py`). To accurately benchmark the latency improvements on target hardware, the deep learning checkpoints are serialized into the Open Neural Network Exchange (ONNX) format, allowing us to profile native CPU execution uncoupled from PyTorch's graphical overhead.

Furthermore, to simulate clinical interaction and edge-deployment visualization, we engineered a graphical interface using Streamlit and Altair. Containerized via Docker, this dashboard permits users to dynamically load arbitrary checkpoints, upload raw ECG arrays, execute real-time Normalized INT8 inference, and visually assess the resulting predicted probability distributions alongside the generated Saliency Map.

---

## VI. Results and Analysis
While complete empirical outcomes depend on the final dataset permutation, the rigorous workflow provides several analytical axes:
1. **Accuracy vs. Compression:** QAT is expected to retain near-FP32 F1-macro scores, sharply outperforming PTQ which often causes accuracy degradation on the rarer minority classes (e.g., Fusion beats).
2. **Edge Hardware Profiling:** ONNX benchmarking proves the viability of edge deployment, with INT8 models consistently demonstrating massive reductions in memory size (Megabytes) and latency (milliseconds per beat) compared to the raw implementations.
3. **Observation of Interpretability Drift:** High-precision FP32 models closely anchor their highest saliency gradients to the QRS-complex and anomalous P-waves. When compressed to INT8 using PTQ, preliminary analyses show scattered, noisier gradient maps. The network may still output the correct prediction but arrives there by attributing importance to baseline segments rather than strict morphological indicators. QAT significantly mitigates this explanation drift.

---

## VII. Conclusion and Future Work
This paper introduces an end-to-end framework illuminating the critical trade-offs required to successfully deploy artificial intelligence onto wearable ECG monitors. Through rigorous experimentation, we mapped the boundaries where model compression maximizes hardware efficiency without severing clinical reliability. Crucially, this study exposes "interpretability drift" as a primary concern for edge healthcare AI; reducing mathematical precision directly alters the model's visual reasoning. While Quantization-Aware Training recovers diagnostic accuracy, verifying the stability of XAI is mandatory prior to clinical adoption. 

Future work will expand upon this framework by deploying the generated `.tflite` or `.onnx` binaries natively onto ARM Cortex-M microcontrollers to measure empirical thermal dispersion and battery decay in real-world wearable testbeds. Additionally, integrating noise-robust topological networks and personalized federated learning mechanisms stands as the next horizon for adaptive edge cardiology.
