# SMS Spam Classifier

A machine learning project that classifies SMS messages as spam or ham (legitimate) using Natural Language Processing and Naive Bayes algorithm.

## Features

- **Text Classification**: Accurately classifies SMS messages as spam or ham
- **TF-IDF Vectorization**: Converts text data into numerical features
- **Naive Bayes Model**: Uses MultinomialNB for efficient text classification
- **Web Interface**: Easy-to-use web UI for real-time message classification
- **Model Persistence**: Saves trained model and vectorizer for reuse

## Dataset

The project uses the SMS Spam Collection dataset, which contains:
- 5,574 SMS messages
- Two categories: 'ham' (legitimate) and 'spam'
- Tab-separated format with labels and message content

## Project Structure

```
sms-spam-classifier/
│
├──model.py                      # Main training script
├── app.py                       # Flask web application
├── spam_classifier_model.pkl    # Trained model (generated)
├── tfidf_vectorizer.pkl         # TF-IDF vectorizer (generated)
├── SMSSpamCollection            # Dataset file
├── templates/
    └── index.html              # Web UI template

```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install pandas scikit-learn flask joblib
```

Or create a requirements.txt file:

```txt
pandas==1.5.3
scikit-learn==1.3.0
flask==2.3.3
joblib==1.3.2
```

Then install:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

First, ensure you have the `SMSSpamCollection` dataset file in your project directory, then run:

```bash
python spam_classifier.py
```

This will:
- Load and preprocess the dataset
- Split data into training and testing sets
- Train the Naive Bayes classifier
- Display classification metrics
- Save the trained model and vectorizer

### 2. Run the Web Application

Run the Flask web application (`app.py`):


### 3. Create Web UI Templates

Run the directory structure and HTML template:


### 4. Start the Web Application

```bash
python app.py
```

Visit `http://localhost:5000` in your browser to use the web interface.

## Model Performance

The classifier typically achieves:
- **Accuracy**: ~97-98%
- **Precision**: High for both spam and ham detection
- **Recall**: Balanced performance across classes

## Technical Details

### Algorithm
- **Classifier**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Preprocessing**: English stop words removal

### Features
- **Text Processing**: Automatic handling of text preprocessing
- **Feature Engineering**: TF-IDF converts text to numerical features
- **Model Persistence**: Trained models saved for production use


## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request



---

**Note**: This classifier is for educational purposes. For production use, consider additional security measures and more sophisticated preprocessing techniques.
