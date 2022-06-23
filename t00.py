import flask

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
    print(file)
    return flask.render_template("aa.html")

app.run(debug=True, host='0.0.0.0', port=8888)