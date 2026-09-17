# Vision Model Dossier

## Purpose
Responsible for OCR/text recognition, image classification, object detection/selection, and visual CAPTCHA-style challenges.

## Architecture
*(To be determined during implementation - modular backbone with multiple heads expected)*

## Datasets
- **MJSynth / Synth90k**: Large-scale synthetic word recognition.
- **SynthText**: Synthetic text embedded in natural scenes.
- **IIIT5K**: Small real-world OCR dataset.
- **TextOCR**: Real-world scene text.
- **MS COCO**: Object detection and classification.
- **Generated**: Synthetic CAPTCHA challenges.

## Training Strategy
Curriculum staging: 
Foundation Visual → Synthetic OCR → Real OCR → Object Training → CAPTCHA Fine-Tuning → Robustness Training

## Evaluation Metrics
- **OCR**: CER, WER, exact-match accuracy
- **Detection**: mAP, IoU, precision, recall
- **Classification**: Accuracy, precision, recall, F1

## Current Results
**Not evaluated**
