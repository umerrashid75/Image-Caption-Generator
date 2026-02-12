# Neural Storyteller

> **"An AI that Sees and Speaks"**

**Neural Storyteller** is an advanced image captioning application that leverages deep learning to generate descriptive captions for any uploaded image. Built with a **ResNet-50 Encoder** and an **LSTM Decoder**, it bridges the gap between Computer Vision and Natural Language Processing. The project features a premium, SaaS-style dark interface built with **Streamlit**.

---

## Features

-   **Deep Learning Core**: Utilizes a pre-trained **ResNet-50** (ImageNet weights) for feature extraction and a custom **LSTM** for sequence generation.
-   **Decoder Strategies**: Implements both **Greedy Search** (fast) and **Beam Search** (high accuracy) for caption generation.
-   **Premium UI**: A custom-styled Streamlit interface with:
    -   Pure black minimal aesthetic.
    -   Glassmorphism effects and Teal Blue accents.
    -   Seamless no-scroll layout and animated components.
-   **Real-time Processing**: Fast inference on CPU or GPU.

## Tech Stack

*   **Core**: Python
*   **Deep Learning**: PyTorch, Torchvision
*   **Web Framework**: Streamlit
*   **NLP**: NLTK (Tokenizer)
*   **Image Processing**: Pillow (PIL)
*   **Data Handling**: Pandas, Numpy

## Project Structure

```bash
Neural-Storyteller/
├── app.py                 # Main Streamlit application (Frontend)
├── model_architecture.py  # PyTorch model definitions (Encoder, Decoder, Seq2Seq)
├── neural_storyteller.pth # Trained model weights
├── vocab.pkl              # Pickled vocabulary file
├── requirements.txt       # Project dependencies
└── README.md              # Documentation
```

## Installation

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

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Model Architecture

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
