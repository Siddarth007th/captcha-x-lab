# Shared ML Utilities Dossier

## Purpose
This folder holds code that is *genuinely* shared by all four model systems (Vision, Reasoning, Audio, Master).

## Guidelines
- **DO NOT** use this as a dumping ground.
- Model-specific implementations must remain inside that model's folder.

## Contents
- **`interfaces/`**: Common abstract base classes and typed interfaces (e.g., `BaseEvaluator`).
- **`evaluation/`**: Common metric calculation utilities and logging frameworks (e.g., `ExperimentTracker`).
- **`preprocessing/`**: Generic data transformations applied universally.
- **`utilities/`**: Seed setting, logging utilities, and mixture sampling algorithms (`DatasetMixtureManager`).
