from flask import Flask,render_template,request 
import pickle
import logging

#Loading the model and cv pickle files.
count_vectors = pickle.load(open('cv.pkl','rb')) ##loading cv
spam_classifier = pickle.load(open('spam.pkl','rb')) ##loading model


app = Flask(__name__) ## defining flask name

@app.route('/') ## home route
def home():
    logging.basicConfig(filename='The_Logs/log_files.log',
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(module)s---- %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger('')
    f = open('The_Logs/log_files.log', 'w+')
    f.truncate()
    return render_template('home.html') ##at home route returning index.html to show

@app.route('/predict',methods=['POST']) ## on post request /predict 
def predict():
    if request.method=='POST':     
        mail = request.form['message']  ## requesting the content of the text field
        data = [mail] ## converting text into a list
        vect = count_vectors.transform(data).toarray() ## transforming the list of sentence into vecotor form
        pred = spam_classifier.predict(vect) ## predicting the class(1=spam,0=ham)
        return render_template('result.html',prediction=pred) ## returning result.html with prediction var value as class value(0,1)
if __name__ == "__main__":
    app.run(debug=True)
