#FLASK
from flask import Flask, render_template,request
import joblib

app = Flask(__name__)
# ['Attr16', 'Attr25', 'Attr26', 'Attr34', 'Attr41']
@app.route("/", methods=["GET", "POST"]) #even though only using post, best practice to have both
def index():
  if request.method == "POST":
        attr16 = request.form.get("attr16")
        attr25 = request.form.get("attr25")
        attr26 = request.form.get("attr26")
        attr34 = request.form.get("attr34")
        attr41 = request.form.get("attr41")
        model_lr = joblib.load("BN_LR")
        pred_lr = model_lr.predict([[attr16,attr25,attr26,attr34,attr41]])
        model_svm = joblib.load("BN_SVM")
        pred_svm = model_svm.predict([[attr16,attr25,attr26,attr34,attr41]])
        model_knn = joblib.load("BN_KNN")
        pred_knn = model_knn.predict([[attr16,attr25,attr26,attr34,attr41]])
        str1 = "The prediction for bankruptcy is " + str(pred_lr) + " using logistic regression, " + str(pred_svm) + " using SVM and " + str(pred_knn) + " using KNN. (1 means bankruptcy)"
        return (render_template("index.html",result1=str1))
  else:
    return(render_template("index.html", result1=".."))

if __name__=="__main__": 
  app.run()