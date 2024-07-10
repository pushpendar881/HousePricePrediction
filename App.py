from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaler1.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Convert generator to list of floats
    features = [float(x) for x in request.form.values()]
    final_features = scaler.transform(np.array(features).reshape(1, -1))
    print(final_features)
    result = regmodel.predict(final_features)[0]
    return render_template('index.html', prediction_text="The predicted value of house is ${:.2f} ".format(result))

if __name__ == "__main__":
    app.run(debug=True)
