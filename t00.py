import flask
import base64
import numpy as np
import cv2
app = flask.Flask(__name__)

@app.route("/")
def get():
    return flask.render_template("index.html")

@app.route("/aa")
def getaa():
    return flask.render_template("aa.html")

@app.route("/img",methods=["POST"])
def getaimg():
    file = flask.request.form

    img = cv2.imdecode(np.frombuffer(base64.b64decode(file['singlefile'].split(',')[1]),np.uint8),cv2.IMREAD_COLOR)

    cv2.imshow('',img)
    cv2.waitKey()
    return flask.render_template("aa.html")

app.run(debug=True, host='0.0.0.0', port=8888)
