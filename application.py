from flask import Flask, render_template, request
import json
import csv
import sys
application = Flask(__name__)
data=["223","1231"]

@application.route("/index")
def index():
    return render_template("index.html")

#예측
@application.route("/search")
def search():
    return render_template("search.html")

@application.route("/result", methods=["POST","GET"])
def result():
    if request.method == "POST":
        result = request.form
        return render_template("result.html",result=result)


if __name__ == "__main__":
    application.run(host='0.0.0.0')