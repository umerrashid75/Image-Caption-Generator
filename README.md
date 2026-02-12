# 🧠 Neural Storyteller

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **"An AI that Sees and Speaks"**

**Neural Storyteller** is an advanced image captioning application that leverages deep learning to generate descriptive captions for any uploaded image. Built with a **ResNet-50 Encoder** and an **LSTM Decoder**, it bridges the gap between Computer Vision and Natural Language Processing. The project features a premium, SaaS-style dark interface built with **Streamlit**.

---

## ✨ Features

-   **Deep Learning Core**: Utilizes a pre-trained **ResNet-50** (ImageNet weights) for feature extraction and a custom **LSTM** for sequence generation.
-   **Decoder Strategies**: Implements both **Greedy Search** (fast) and **Beam Search** (high accuracy) for caption generation.
-   **Premium UI**: A custom-styled Streamlit interface with:
    -   Pure black (`#000000`) minimal aesthetic.
    -   Glassmorphism effects and Teal Blue (`#14b8a6`) accents.
    -   Seamless no-scroll layout and animated components.
-   **Real-time Processing**: Fast inference on CPU or GPU.

## 🛠️ Tech Stack

*   **Core**: Python 3.12
*   **Deep Learning**: PyTorch, Torchvision
*   **Web Framework**: Streamlit
*   **NLP**: NLTK (Tokenizer)
*   **Image Processing**: Pillow (PIL)
*   **Data Handling**: Pandas, Numpy

## 📂 Project Structure

```bash
📂 Neural-Storyteller/
├── app.py                 # Main Streamlit application (Frontend)
├── model_architecture.py  # PyTorch model definitions (Encoder, Decoder, Seq2Seq)
├── neural_storyteller.pth # Trained model weights
├── vocab.pkl              # Pickled vocabulary file
├── requirements.txt       # Project dependencies
└── README.md              # Documentation
```

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/neural-storyteller.git
    cd neural-storyteller
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download NLTK data** (automatically handled in app, but good to know):
    ```python
    import nltk
    nltk.download('punkt')
    ```

4.  **Ensure Model Files**:
    Place your trained `neural_storyteller.pth` and `vocab.pkl` files in the root directory.

## 🎮 Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

## 🧠 Model Architecture

The model follows a classic **Show and Tell** architecture:

1.  **Encoder (ResNet-50)**:
    -   Takes an input image (resized to 224x224).
    -   Passes it through a ResNet-50 CNN (last layer removed).
    -   Outputs a feature vector of size `2048`.
    -   Projects this vector to the embedding dimension (`256`).

2.  **Decoder (LSTM)**:
    -   Initialized with the image features.
    -   Receives word embeddings step-by-step.
    -   Predicts the next word probability distribution over the vocabulary.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ for GenAI Assignment
</p>
