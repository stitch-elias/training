import flask
import base64
import numpy as np
import cv2
import onnxruntime
import os
app = flask.Flask(__name__)
app.secret_key='any random string'
app.config['MAX_CONTENT_LENGTH'] = 1024*1024*50
app.config['DEBUG'] = True

if os.path.exists(""):
    db = np.load("",allow_pickle=True).item()
else:
    db = {}

@app.route("/")
def get():
    username = flask.request.cookies.get("id")
    if username!=None:
        if "id" not in flask.session.keys() and flask.session["state"]=="login":
            flask.session["id"]=username
    return flask.render_template("index.html")

@app.route("/aa")
def getaa():
    if "id" in flask.session.keys() and flask.session["state"] == "login":
        return flask.render_template("aa.html")
    else:
        return flask.redirect(flask.url_for("get"))


@app.route("/bb")
def getbb():
    if "id" in flask.session.keys() and flask.session["state"] == "login":
        return flask.render_template("bb.html")
    else:
        return flask.redirect(flask.url_for("get"))

@app.route("/cc")
def getcc():
    if "id" in flask.session.keys() and flask.session["state"] == "login":
        return flask.render_template("cc.html")
    else:
        return flask.redirect(flask.url_for("get"))

@app.route("/upload", methods=["POST"])
def upload_file():
    t = flask.request.files.getlist("imgFile")[0].filename.split(".")[-1]
    buffer = flask.request.files.getlist("imgFile")[0].stream.read()

    if t in ["jpg","png","jpeg"]:
        img = cv2.imdecode(np.frombuffer(buffer,np.uint8),cv2.IMREAD_COLOR)
        img = parsing(img)
        _, img_ = cv2.imencode(".png", img)
        b64 = base64.b64encode(img_.tobytes()).decode("utf-8")
    else:
        b64=""
    return b64


@app.route("/img",methods=["POST"])
def getaimg():
    file = flask.request.form
    print(file['singlefile'])
    img = cv2.imdecode(np.frombuffer(base64.b64decode(file['singlefile'].split(',')[1]),np.uint8),cv2.IMREAD_COLOR)
    _,img_ = cv2.imencode(".png",img)
    b64 = base64.b64encode(img_.tobytes()).decode("utf-8")
    return b64

@app.route("/login",methods=["POST"])
def login():
    name = flask.request.form["n"]
    password = flask.request.form["p"]
    response = flask.redirect(flask.url_for("get"))
    if name in db.keys() and db[name]==password:
        flask.session["id"]=name
        flask.session["state"]="login"
        response.set_cookie("id", name)
    return response

@app.route("/logup",methods=["POST"])
def logup():
    name = flask.request.form["n"]
    password = flask.request.form["p"]
    response = flask.redirect(flask.url_for("get"))
    db[name]=password
    flask.session["id"]=name
    flask.session["state"]="login"
    response.set_cookie("id", name)
    return response

@app.route("/logout")
def logout():
    flask.session.pop("id")
    flask.session["state"]="logout"
    return flask.redirect(flask.url_for("get"))

def parsing(img):
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h,w = img.shape[:-1]
    delt = np.max([h,w])
    if h!=w:
        back = np.ones((delt,delt,3),np.uint8)*175
        back[:img.shape[0],:img.shape[1],:]=img
        img = back
    img = cv2.resize(img,(512,512))
    ii = img
    img = np.transpose(img,(2,0,1))
    img = img/255.
    img = img-np.array([0.485, 0.456, 0.406]).reshape((-1,1,1))
    img = img/np.array([0.229, 0.224, 0.225]).reshape((-1,1,1))
    img = img[None,:,:,:]
    img = np.asarray(img,dtype=np.float32)

    session = onnxruntime.InferenceSession("parse.onnx")
    input_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    pred_onx = session.run([out_name], {input_name: img})
    mask = np.argmax(pred_onx[0][0],axis=0)
    # mask0 = np.where(mask<14,1,0)
    mask1 = np.where(mask>0,1,0)
    # mask2 = np.where(mask==17,1,0)
    # m = (mask0+mask2)*mask1
    ii_ = np.asarray(ii*mask1[:,:,None],np.uint8)
    ii_ = cv2.resize(ii_,(delt,delt))
    i_ = ii_[:h,:w]
    cv2.imwrite("2.png",i_)
    return i_
# parsing()

# @app.route("/logout")
# def logout():
#
#     return flask.redirect(flask.url_for("get"))

app.run(debug=True, host='0.0.0.0', port=8888)
