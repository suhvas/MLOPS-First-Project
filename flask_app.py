#import model # Import the python file containing the ML model
import pickle
import numpy as np
from flask import Flask, request, render_template,jsonify # Import flask libraries



# Initialize the flask class and specify the templates directory
app = Flask(__name__,template_folder="templates")


#model = pickle.load(open('model.pkl','rb'))

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# Default route set as 'home'
@app.route('/')
def home():
    return render_template('home.html') # Render home.html


@app.route('/classify', methods=['POST','GET'])
def classify():
    # Retrieve the query parameters from the request
    slen = float(request.args.get('slen'))
    swid = float(request.args.get('swid'))
    plen = float(request.args.get('plen'))
    pwid = float(request.args.get('pwid'))

    # Create a new data point for prediction
    data = [[slen, swid, plen, pwid]]

    # Perform prediction using the trained classifier
    prediction = model.predict(data)
    output =str(prediction[0])

    return render_template('output.html', prediction_text='The Flower is {}'.format(output))

# Run the Flask server
if(__name__=='__main__'):
    app.run(debug=True)        