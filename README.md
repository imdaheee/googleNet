# GoogLeNet from Scratch

This project implements **GoogLeNet (Inception v1)** from scratch using PyTorch. The goal is to understand and reproduce the original architecture without relying on pre‑trained models or torchvision’s built‑in implementation.

## 1. Dataset

The model is trained and evaluated on **POC_Dataset**, which contains four pathological tissue classes:

* Chorionic_villi
* Decidual_tissue
* Hemorrhage
* Trophoblastic_tissue

Data is split as follows:

* **Training**: Full Training folder
* **Validation**: 80% of Testing folder
* **Testing**: Remaining 20% of Testing folder

## 2. Training Setup

| 항목         | 설정                                             |
| ---------- | ---------------------------------------------- |
| Input Size | 224×224                                        |
| Batch Size | 8                                              |
| Optimizer  | SGD (lr=0.01, momentum=0.9, weight decay=5e-4) |
| Epochs     | 10                                             |
| Loss       | CrossEntropy + Auxiliary loss (×0.3 each)      |
| Device     | CPU / Apple MPS (Auto)                         |

## 3. Model Architecture

GoogLeNet consists of:

* Stem convolution layers
* Inception modules (3a–3b, 4a–4e, 5a–5b)
* Auxiliary classifiers (at 4a, 4d)
* Global average pooling + final FC layer

## 4. Training Results

Below is a summary of the training outcome:

| Metric                   |  Accuracy  |
| ------------------------ | :--------: |
| Training Accuracy        | **81.52%** |
| Best Validation Accuracy | **81.71%** |
| Test Accuracy            | **81.85%** |

These results indicate that the model generalizes well to unseen data, as the validation and test accuracies are closely aligned.

최종 성능은 다음과 같습니다:

| Metric                   |  Accuracy  |
| ------------------------ | :--------: |
| Train Accuracy           | **81.52%** |
| Best Validation Accuracy | **81.71%** |
| Test Accuracy            | **81.85%** |

Validation과 Test 성능이 유사하여, 모델이 안정적으로 일반화된 것으로 판단할 수 있습니다.

## 5. File Structure

```
googleNet/
├─ models/
│   ├─ __init__.py
│   └─ googlenet.py
├─ train.py
└─ best_googlenet.pth
```

---

다음으로 추가 예정:

* Confusion matrix
* Training/validation accuracy & loss curves
* Inference examples (visualization)
