from flask import Flask, render_template, request
import joblib

# Load the trained model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]
        result = 'Spam' if prediction == 1 else 'Not Spam'
        return render_template('index.html', message=message, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
