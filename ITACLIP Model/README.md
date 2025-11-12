# 🧠 ITACLIP-JAX — Image-Text Alignment Model in JAX

This project implements **ITACLIP**, a lightweight image-text alignment model inspired by CLIP (Contrastive Language–Image Pretraining), built using **JAX** and **Flax**.

---

## 🚀 Overview

ITACLIP-JAX demonstrates how to train and evaluate a small-scale multimodal model that learns to connect images and their textual descriptions.  
The focus is on simplicity, reproducibility, and understanding the fundamentals of contrastive multimodal learning.

---

## 🧩 Features

- Implemented entirely in **JAX** / **Flax**
- Supports both **image encoder (CNN)** and **text encoder (Transformer or MLP)**
- Includes training and evaluation notebooks
- Lightweight and easy to extend

---

## 📁 Folder Structure

```
ITACLIPModel/
│
├── itaclip-jax.ipynb # Main training and demo notebook
└── README.md # Project documentation

```

---

## ⚙️ Requirements

Install dependencies:
```bash
pip install jax jaxlib flax numpy matplotlib tqdm

```
---
Usage

Open the notebook and run all cells to:

Train the ITACLIP model on a small image-text dataset

Visualize the alignment between embeddings

Evaluate similarity using cosine distance

📊 Results

Example output plots show:

Image-text embedding alignment

Training loss over epochs

Cosine similarity heatmap

🔍 References

CLIP: Learning Transferable Visual Models from Natural Language Supervision (Radford et al., 2021)

JAX Documentation

Flax Library
---

```markdown
### 🧩 ITACLIP-JAX — Image-Text Alignment Model
A small JAX/Flax-based implementation of a CLIP-style model that aligns image and text embeddings.
📂 [View Folder](./ITACLIPModel)

```

CLIP: Learning Transferable Visual Models From Natural Language Supervision
Authors: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
Published by: OpenAI (2021)
📄 Paper Link: https://arxiv.org/abs/2103.00020

```
@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  booktitle={Proceedings of the 38th International Conference on Machine Learning},
  year={2021},
  organization={PMLR}
}
```
```
## 🔍 References

- [CLIP: Learning Transferable Visual Models from Natural Language Supervision (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Library](https://flax.readthedocs.io/)

```