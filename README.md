# HW2_MNLP_NonModernIt2ModernIt
This project explores automatic translation of archaic Italian texts into modern Italian, a task relevant to digital humanities, historical linguistics, and cultural preservation.

We experiment with different translation models:
- Flan-T5-XL
- Gemma 3B (various decoding strategies)
- Helsinki-NLP models
- LLaMA 8B

Model outputs are evaluated in two ways:

1. Automated Evaluation with Prometheus

- Implemented in prometheus.py

- Uses the PrometheusEval class to produce fine-grained quality scores

2. Human-Aligned Evaluation

- Conducted in HW2_evaluate.ipynb

- Compares:

    - Human-provided reference scores

    - Prometheus scores

    - ChatGPT evaluations

This allows for analysis of how well automated metrics correlate with human judgment.

Datasets:
    train.csv: Contains training examples in two columns: archaic text and its modern Italian version.
    test.csv: Contains test examples used for evaluating the models' generalization capabilities.