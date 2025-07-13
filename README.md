# AdSift – Sifting Through Ads Intelligently

**AdSift** is a web application built with FastAPI and TensorFlow that uses a pre-trained BERT model to classify text into categories such as Art and Music, Food, History, Manufacturing, Science and Technology, and Travel. The application provides a user-friendly interface to input text, receive classification results, and visualize class probabilities with a bar chart.

![](/images/image.png)

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Text Classification**: Classify input text into one of six predefined categories using a BERT-based model.
- **Interactive Web Interface**: A clean, responsive UI built with HTML, CSS, and JavaScript.
- **Probability Visualization**: Displays classification probabilities as a bar chart using Chart.js.
- **FastAPI Backend**: Efficiently handles API requests for text predictions.
- **Responsive Design**: Optimized for both desktop and mobile devices.

## Tech Stack
- **Backend**: FastAPI (Python) for the API server.
- **Machine Learning**: TensorFlow and Hugging Face Transformers (BERT model).
- **Frontend**: HTML, CSS (with Poppins font), JavaScript, and Chart.js for visualizations.
- **Static Files**: Served via FastAPI's `StaticFiles` for CSS and other assets.
- **Templates**: Jinja2 for rendering HTML templates.
- **Pydantic**: For request body validation.

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- A pre-trained BERT classification model saved as `bert_classification_model`.

### Steps
1. **Clone the Repository**:
   ```
   git clone https://github.com/jarif87/adsift-sifting-through-ads-intelligently.git

   ```
2. **Set Up a Virtual Environment**
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. **Install Dependencies**
```
tensorflow==2.18.0
transformers==4.52.4
seaborn==0.12.2
uvicorn==0.34.3
matplotlib==3.7.2
pandas==2.2.3
fastapi
numpy==1.26.4
tf_keras==2.18.0
Jinja2
python-multipart
```
### Prepare the Model:
- Ensure the pre-trained BERT model (bert_classification_model) is available in the project root directory.
- The model should be compatible with TensorFlow and the Hugging Face TFBertModel.

### Directory Setup:
- Create a static directory and place the style.css file inside it.
- Create a templates directory and place the index.html file inside it.

### Run the Application

```
uvicorn main:app 
```

### Usage
- Access the Web Interface:
    - Open http://localhost:8000 in your browser.
    - Enter text in the provided textarea and click the "Classify Text" button.

### Project Structure

```
adsift/
├── static/
│   └── style.css          # CSS styles for the frontend
├── templates/
│   └── index.html         # HTML template for the web interface
├── main.py                # FastAPI application code
├── bert_classification_model/  # Directory or file for the pre-trained BERT model
└── README.md              # Project documentation
```

### Model Details
- Model: BERT (base-uncased) fine-tuned for text classification.
- Categories: Art and Music, Food, History, Manufacturing, Science and Technology, Travel.
- Input: Text tokenized with a maximum length of 128 tokens.
- Output: Predicted class and probability distribution across all categories.
- Tokenizer: Hugging Face AutoTokenizer for BERT-base-uncased.

