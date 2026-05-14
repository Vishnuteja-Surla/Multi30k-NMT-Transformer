# Assignment 3: Implementing a Transformer for Machine Translation

This directory contains the complete implementation, experiments, and analysis for **Assignment 3** of **DA6401: Introduction to Deep Learning**.

The focus of this assignment is to implement the landmark architecture from the paper *"Attention Is All You Need"* from scratch using PyTorch. Transitioning from convolutional and recurrent networks, this project builds a purely attention-based sequence-to-sequence model capable of Neural Machine Translation (NMT) from German to English using the **Multi30k Dataset**.

For the public Weights & Biases report detailing the analysis of this assignment, click here: [W&B Report Link](https://wandb.ai/cs25m050-indian-institute-of-technology-madras/DA6401_Assignment_03/reports/DA6401-Assignment-3-Report--VmlldzoxNjg3OTEwOA?accessToken=9i4a0kdzt15jllprek1j7qy16sgfemdyr3l9bonymui0i36oe6szd0hn53clow6y)

For the codebase of this current project, click here: [GitHub Link](https://github.com/Vishnuteja-Surla/Multi30k-NMT-Transformer) *(Note: Follows the official Assignment-3 GitHub Skeleton)*

---

## Assignment Objectives

The objectives of this assignment are to:

* **Task 1: Scaled Dot-Product and Multi-Head Attention** Implement the core attention mechanisms from scratch, including strict padding and causal (look-ahead) masking to prevent data leakage from future tokens.
* **Task 2: Transformer Encoder and Decoder Stacks** Construct the full encoder/decoder layers utilizing sinusoidal Positional Encodings, Point-wise Feed-Forward Networks, and "Add & Norm" residual connections.
* **Task 3: Training Pipeline and Optimization** Implement specific training strategies critical for Transformer convergence, including Label Smoothing ($\epsilon_{ls}=0.1$), the Noam Learning Rate Scheduler (with a warmup phase), and a greedy autoregressive decoding function for inference.

---

## Constraints & Academic Integrity

* **Frameworks:** Entire implementation must be in PyTorch using basic building blocks (`nn.Linear`, `nn.Module`).
* **Prohibited Modules:** The use of `torch.nn.MultiheadAttention` is strictly forbidden.
* **Permitted Libraries:** `torch`, `numpy`, `matplotlib`, `scikit-learn`, `wandb`, `datasets`, `spacy`, `bleu`, `tqdm`. All tokenization must utilize `spacy`.
* **AI Usage:** Tools like ChatGPT or Claude are permitted only as conceptual aids; they must not be used to generate the final code submission.
* **Data Integrity:** Training and test datasets must be strictly isolated. Any attempt to artificially inflate accuracy (e.g., data leakage) will result in an immediate grade of zero.
* **Plagiarism:** All submissions will undergo rigorous plagiarism and AI-generated code detection.

---

## Implementation & Architecture Notes

This implementation is designed to satisfy a strict automated evaluation pipeline:

* **Autograder Signatures:** Core functions (`scaled_dot_product_attention`, `MultiHeadAttention.forward`, `make_src_mask`, etc.) adhere to strict unmodifiable signatures for isolated unit testing.
* **Autoregressive Inference Pipeline:** An `infer()` method is implemented directly inside the `Transformer` class to execute end-to-end NLP processing: accepting a raw German string, tokenizing, encoding, decoding step-by-step, and detokenizing to English.
* **Dynamic Weight Loading:** The `Transformer.__init__()` handles dynamic loading of vocabulary mappings and utilizes `gdown` to download the trained weights (`best_model.pth`) directly from Google Drive during inference to satisfy autograder constraints.
* **Automated Evaluation Metric:** Corpus-level BLEU score evaluated on a held-out test set.

---

## Weights & Biases Report

A **public Weights & Biases report** accompanies this assignment and includes rigorous analysis of five specific ablation studies and experiments:

* **The Necessity of the Noam Scheduler:** A comparison of training loss and validation accuracy curves between the Noam Scheduler (warmup + decay) and a fixed learning rate, demonstrating sensitivity to initial learning rates.
* **Ablation: The Scaling Factor:** An analysis logging the gradient norms of the Query and Key weights to empirically demonstrate the "vanishing gradient" problem when the $1/\sqrt{d_k}$ factor is omitted.
* **Attention Rollout & Head Specialization:** Extracted attention heatmaps from the last encoder layer to visualize head specialization and observe potential "Head Redundancy".
* **Positional Encoding vs. Learned Embeddings:** A comparative study between sinusoidal positional encodings and `torch.nn.Embedding`, discussing theoretical sequence length extrapolation.
* **Decoder Sensitivity & Label Smoothing:** Visualizing the "Prediction Confidence" to explain how $\epsilon_{ls}=0.1$ acts as a regularizer to prevent the model from becoming over-confident.

---

## Submission Notes

* A **Public W&B Report** link must be accessible during the evaluation phase; failure to do so will result in a negative marking penalty.
* The formal submission of the codebase and report must be completed via **Gradescope**.
* No extensions will be granted beyond the provided deadline under any circumstances.

---

## Timeline

* **Release Date:** 24th April 2026, 10:00 AM
* **Submission Deadline:** 19th May 2026, 23:59 PM
* **Late Submission Deadline:** 24th May 2026, 23:59 PM (with penalty)
* **Submission Platform:** Gradescope

---

## Relation to Course Objectives

This assignment directly supports the course goals of:

* Shifting from sequential processing architectures (CNNs/RNNs) to highly parallelizable attention mechanisms.
* Implementing state-of-the-art NLP models completely from scratch to understand their internal calculus and matrix dimensionalities.
* Understanding the specific optimization hurdles (vanishing gradients, initialization sensitivity) native to deep Transformer networks.
* Evaluating generative sequence models using industry-standard metrics like BLEU.