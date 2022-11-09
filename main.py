from flask import Flask,render_template,request 
import pickle

###Loading model and cv
cv = pickle.load(open('/config/workspace/Spam_Ham_Classifier/vectors.pkl','rb')) ##loading cv
model = pickle.load(open('/config/workspace/Spam_Ham_Classifier/spam_classifier_model.pkl','rb')) ##loading model

app = Flask(__name__) ## defining flask name

@app.route('/') ## home route
def home():
    return render_template('home.html') ##at home route returning index.html to show

@app.route('/predict',methods=['POST']) ## on post request /predict 
def predict():
    if request.method=='POST':     
        mail = request.form['email']  ## requesting the content of the text field
        data = [mail] ## converting text into a list
        vect = cv.transform(data).toarray() ## transforming the list of sentence into vecotor form
        pred = model.predict(vect) ## predicting the class(1=spam,0=ham)
        return render_template('result.html',prediction=pred) ## returning result.html with prediction var value as class value(0,1)
if __name__ == "__main__":
    app.run(debug=True)     ## running the flask app as debug==True
