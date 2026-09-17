# Reasoning Model Dossier

## Purpose
Vision-Language model responsible for spatial reasoning, counting, object relationships, compositional reasoning, visual question answering, and complex visual challenges.

## Architecture
*(To be determined - likely a VLM combining a vision encoder and language/reasoning module)*

## Datasets
- **COCO**: General visual grounding.
- **Visual Genome**: Objects, attributes, regions, and relationships.
- **VQA v2**: Visual question answering.
- **GQA**: Compositional and spatial visual reasoning.
- **Generated**: Controlled reasoning challenges.

## Training Strategy
Curriculum staging:
Visual Foundation → Visual Relationships → VQA → Compositional Reasoning → Controlled Reasoning Challenges

## Evaluation Metrics
- Exact-match accuracy
- Category-level accuracy
- OOD accuracy

## Current Results
**Not evaluated**
