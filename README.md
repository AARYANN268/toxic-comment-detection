# toxic-comment-classification
A multi-label classification project using Deep Learning (TensorFlow) to detect various types of toxicity in online comments, including threats, insults, and identity hate
This project implements a multi-label text classification model to detect various forms of toxicity in social media comments. The model is capable of identifying multiple labels for a single comment, such as **toxic, severe_toxic, obscene, threat, insult, and identity_hate**.
## 🚀 Key Features
- **Data Preprocessing**: Custom text cleaning pipeline using Regular Expressions to remove punctuation, newlines, and non-alphabetic characters.
- **Text Vectorization**: Utilizes the `TextVectorization` layer from TensorFlow with a vocabulary size of **150,000** and a sequence length of **1,200**.
- **Deep Learning Model**: Developed using TensorFlow and Keras with a focus on sequence modeling.
- **Efficient Pipelines**: Implements `tf.data.Dataset` for high-performance data loading, including caching, shuffling, and prefetching.
- **GPU Acceleration**: Configured to run on Google Colab with NVIDIA T4 GPU support.
## 🛠️ Tech Stack
- **Languages**: Python
- **Deep Learning**: TensorFlow, Keras
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
## 📊 Dataset & Analysis
The project analyzes a dataset containing over **159,000** comments. 
- It includes an Exploratory Data Analysis (EDA) phase to visualize the distribution of toxicity labels.
- The training process uses an **80/20 train-validation split**.
## ⚙️ Usage
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy tensorflow matplotlib seaborn
