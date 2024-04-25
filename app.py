from flask import Flask ,render_template,request
import pickle
import joblib
import numpy as np
app = Flask(__name__)
model=joblib.load("student_marks_predictor_model (1).pkl")

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict_marks():
    hours = float(request.form.get('study_hours'))
    hours_value = np.array(hours)

    #prediction
    result = model.predict([[hours_value]])
    return render_template('index.html',result='{}marks'.format(result))
if __name__ == '__main__':
    app.run(debug=True)
