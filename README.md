# ECA-GRAD: Two-Tier AI-Agent Framework for Breast Cancer Histopathology Classification

An academic research prototype for **benign/malignant breast histopathology image classification** using a custom Efficient Channel Attention Convolutional Neural Network (ECA-CNN), visual explanations, and a rule-based supervisory decision layer.

> **Research prototype only.** This project is not a clinically validated diagnostic system, does not prescribe treatment, and must not be used to make patient-care decisions. Predictions and visual explanations require independent validation by qualified experts before any clinical application.

## Project Overview

Breast histopathology images contain visual patterns that can help distinguish benign from malignant tissue. This project explores a deep-learning workflow that combines image classification with explainability and a supervisory review mechanism.

The implementation is organized as a two-tier pipeline:

### Tier 1 — Image Classification and Explainability

- Uses the BreakHis breast histopathology image dataset.
- Trains and evaluates a plain CNN as an internal comparator.
- Implements a custom CNN incorporating **Efficient Channel Attention (ECA)** to emphasize informative feature channels.
- Produces a binary prediction: **Benign** or **Malignant**.
- Uses **Grad-CAM** to visualize image regions that contribute to the model's malignant-class score.
- Includes an optional **SHAP GradientExplainer** analysis cell for selected held-out images, subject to successful installation and execution.

### Tier 2 — Supervisory Decision Agent

A separate rule-based supervisory component receives the model's predicted malignant probability and the availability of an explanation. It applies confidence and explanation-availability rules to determine whether the output should be passed through or flagged for human review.

This component is **rule-based**, not a separately trained AI model or an autonomous clinical agent. Its output is intended as an auditable research demonstration, not a diagnosis or treatment recommendation.

## Main Objectives

1. Build a reproducible image-classification workflow for benign and malignant histopathology images.
2. Compare a plain CNN with a custom ECA-CNN under a common experimental setup.
3. Use patient-wise data partitioning to reduce the risk of patient-level data leakage.
4. Evaluate performance using classification, discrimination, and calibration-related measures.
5. Generate Grad-CAM visual explanations and optionally investigate SHAP attributions.
6. Demonstrate a rule-based supervisory layer that can flag uncertain or insufficiently explained predictions for human review.

## Dataset

This project uses the **Breast Cancer Histopathological Image (BreakHis)** dataset.

- **Official dataset:** https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
- **Original dataset paper:** F. A. Spanhol et al., “A dataset for breast cancer histopathological image classification,” *IEEE Transactions on Biomedical Engineering*, vol. 63, no. 7, pp. 1455–1462, 2016, doi: 10.1109/TBME.2015.2496264.

Download the dataset from its official source and review the applicable terms of use. **Do not commit the BreakHis images or redistribute them through this repository unless the dataset terms explicitly permit it.**

### Expected Data Organization

The notebook searches recursively below the dataset root and infers the class from directory names. The extracted dataset should retain its benign/malignant folder structure. The exact folder layout can vary by download package; inspect your copy before running the notebook.

In the notebook's dataset-configuration cell, update `DATA_ROOT` to the location of the extracted `breast` directory on your own computer. Do not commit a machine-specific absolute path to the repository.

The notebook extracts candidate patient identifiers and magnification levels from common BreakHis filename patterns. **Verify the patient-ID parsing against the dataset documentation and actual filenames before relying on patient-wise evaluation.**

## Methodology

1. **Dataset discovery and metadata extraction:** Locate image files and construct a manifest containing image path, class label, candidate patient identifier, and magnification.
2. **Data validation:** Inspect class counts, missing metadata, duplicate paths, file existence, and patient-group consistency.
3. **Patient-wise partitioning:** Create training, validation, and test partitions using patient groups so that images from a patient are not intentionally distributed across partitions. Verify group overlap and class distributions.
4. **Preprocessing:** Resize images and apply the transformations defined in the notebook.
5. **Model development:** Train a plain CNN comparator and the custom ECA-CNN using the same main experiment settings.
6. **Evaluation:** Calculate classification metrics and generate ROC, precision-recall, and calibration-related plots where supported by the executed notebook.
7. **Explainability:** Generate Grad-CAM visualizations; run the optional SHAP cell only if SHAP is installed and the cell executes successfully.
8. **Supervisory review:** Pass the prediction and explanation-availability information to the rule-based Tier-2 component.

## System Architecture

```text
              BreakHis Histopathology Images
                           |
                           v
                 Dataset Preprocessing
                           |
                           v
                 Patient-Wise Splitting
                           |
                           v
             +---------------------------+
             |         TIER 1            |
             |                           |
             |       Plain CNN           |
             |           |               |
             |       ECA-CNN             |
             |           |               |
             |    Classification         |
             |           |               |
             |   Grad-CAM / SHAP         |
             +---------------------------+
                           |
                           v
             Prediction + Confidence
                + Explanation Status
                           |
                           v
             +---------------------------+
             |         TIER 2            |
             |                           |
             |  Rule-Based Supervisor    |
             |                           |
             | Confidence Assessment     |
             | Explanation Availability  |
             | Human Review Flag         |
             +---------------------------+
                           |
                           v
              Structured Decision Output
                 for Human Review
```

*Conceptual representation of the two-tier research architecture.*

## Repository Contents

The intended repository structure is:

```text
ECA-GRAD-Breast-Cancer/
|
|-- README.md
|-- .gitignore
|-- requirements.txt
|-- ECA_GRAD_AI_DA_Updated.ipynb
|
|-- docs/
|   |-- Project_Report.pdf
|   |-- Architecture.png
|
|-- results/
|   |-- evaluation_metrics/
|   |-- confusion_matrix/
|   |-- gradcam_outputs/
|
```

The `docs/` and `results/` folders are optional. Add them when those files are ready.

Do not add the raw BreakHis dataset, private data, local absolute paths, or large model checkpoints to the repository.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/imAditya21/ECA-GRAD-Breast-Cancer.git
cd ECA-GRAD-Breast-Cancer
```

### 2. Create a Virtual Environment

For Windows Command Prompt:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 3. Install PyTorch and Torchvision

Install a compatible **PyTorch** and **Torchvision** build for your operating system and hardware using the official installation selector:

https://pytorch.org/get-started/locally/

The correct installation command depends on whether you use CPU or a supported NVIDIA CUDA configuration.

Verify the installation:

```cmd
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 4. Install Project Dependencies

```cmd
python -m pip install -r requirements.txt
```

### 5. Launch JupyterLab

```cmd
python -m jupyter lab
```

Open `ECA_GRAD_Breast_Cancer_Classification.ipynb` in JupyterLab.

## Running the Notebook

1. Download and extract BreakHis from the official dataset source.
2. Open the notebook and update `DATA_ROOT` in the dataset-configuration cell.
3. Check that the manifest discovers the expected images and that benign/malignant labels are correct.
4. Verify the patient-ID parser and ensure there is no patient overlap between training, validation, and test partitions.
5. Run the notebook from the first cell in order.
6. If the kernel is restarted, rerun the required cells so imports, variables, models, and data loaders are recreated or loaded from saved checkpoints.
7. Review all evaluation outputs and warnings. Do not report metrics until the corresponding cells have completed successfully.
8. Run the SHAP cell only after confirming that `import shap` works in the same Python environment used by the Jupyter kernel.

## Evaluation Metrics

The notebook includes or is designed to calculate:

- Accuracy
- Precision
- Sensitivity (Recall)
- Specificity
- F1-score
- Area Under the ROC Curve (AUROC)
- Area Under the Precision-Recall Curve (AUPRC)
- Brier score
- Expected Calibration Error (ECE)
- Patient-cluster bootstrap confidence intervals for selected ECA-CNN metrics
- Magnification-wise evaluation where the required metadata is available

Grad-CAM visualizations and optional SHAP attributions are also included in the explainability workflow.

**Note:** Metrics, confidence intervals, and plots are generated only when the relevant notebook cells execute successfully. Patient-level bootstrap estimates may be unstable when the number of independent patients is limited. Report the number of valid bootstrap replicates and the grouping method.

No performance values are stated here because they must be taken from a successful, reproducible run of the notebook.

## Important Implementation Notes and Limitations

- The ECA-CNN is a custom implementation; it should not be described as an exact reproduction of ECSAnet from Aldakhil et al. (2024). ECA and CBAM are different attention mechanisms.
- The plain CNN is an internal comparator in this notebook.
- The Tier-2 supervisory component is rule-based and uses predefined logic; it is not a separately trained model.
- Grad-CAM and SHAP are model-explanation techniques, not proof that a prediction is correct or clinically meaningful.
- Patient-wise evaluation depends on correct patient-ID extraction. Validate the parser and inspect group overlap before interpreting results.
- The dataset contains multiple images per patient; image counts and patient counts should be reported separately.
- External validation, broader clinical assessment, and expert review are outside the scope of this prototype.

## Ethics, Privacy, and Responsible Use

- Confirm the dataset's applicable licence and terms before sharing any data or derived artifacts.
- Do not publish patient-identifiable information, raw patient records, private credentials, or local paths containing personal usernames.
- Obtain and report the appropriate institutional ethics determination or approval status. Do not invent an approval or exemption number.
- Consider applicable institutional requirements and relevant Indian guidance, including the Digital Personal Data Protection Act, 2023, and ICMR's *Ethical Guidelines for Application of Artificial Intelligence in Biomedical Research and Healthcare* (2023). Mentioning these sources does not certify legal or ethical compliance.
- Disclose the use of AI tools for coding, debugging, or manuscript preparation in accordance with institutional rules.

## References

1. F. A. Spanhol, L. S. Oliveira, C. Petitjean, and L. Heutte, “A dataset for breast cancer histopathological image classification,” *IEEE Transactions on Biomedical Engineering*, vol. 63, no. 7, pp. 1455–1462, 2016, doi: 10.1109/TBME.2015.2496264.
2. Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, and Q. Hu, “ECA-Net: Efficient channel attention for deep convolutional neural networks,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2020, pp. 11531–11539, doi: 10.1109/CVPR42600.2020.01155.
3. R. R. Selvaraju et al., “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in *Proceedings of the IEEE International Conference on Computer Vision*, 2017, pp. 618–626, doi: 10.1109/ICCV.2017.74.
4. S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Advances in Neural Information Processing Systems*, vol. 30, 2017.
5. L. A. Aldakhil et al., “Attention-based deep learning approach for breast cancer histopathological image multi-classification,” *Diagnostics*, vol. 14, no. 13, Art. no. 1402, 2024, doi: 10.3390/diagnostics14131402.

## Acknowledgement and Project Status

This repository is intended for academic research and educational use. Update this README with the final team details, institutional information, verified dataset and ethics statements, exact experiment configuration, and actual results before submission.
