from distutils.log import debug
from email.mime import audio
from flask import Flask, g, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
import torch
from vncorenlp import VnCoreNLP
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from format import get_correct
from werkzeug.utils import secure_filename
import os
from PIL import Image
from detect import Detect
from PIL import Image
import pytesseract

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD = "./upload"
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
UPLOAD_IMA = "./image"
UPLOAD_IMAGE = os.path.join(APP_ROOT, UPLOAD_IMA)
app.config["UPLOAD_IMAGE"] = UPLOAD_IMAGE
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///mydb.db"
db = SQLAlchemy(app)
tokenizer = AutoTokenizer.from_pretrained("./SC_checking/content/checkpoint-18500")
model = AutoModelForSeq2SeqLM.from_pretrained("./SC_checking/content/checkpoint-18500")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model_checkpoint = "./qs_model/check-point"
nlp = pipeline("question-answering", model=model_checkpoint, tokenizer=model_checkpoint)
d = Detect()
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\kumax\.conda\envs\caps\Library\bin\tesseract.exe"
)


def text(input_text):
    with torch.no_grad():
        tokenized_text = tokenizer(
            input_text, truncation=True, padding=True, return_tensors="pt"
        )

        source_ids = tokenized_text["input_ids"].to(device, dtype=torch.long)
        source_mask = tokenized_text["attention_mask"].to(device, dtype=torch.long)

        generated_ids = model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_length=512,
            num_beams=5,
            repetition_penalty=1,
            length_penalty=1,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
        pred = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    return pred


class Users(db.Model):
    ID_User = db.Column(db.Integer, primary_key=True)
    User_Name = db.Column(db.String(200), nullable=False)
    Email = db.Column(db.String(200), nullable=False)
    Password = db.Column(db.String(200), nullable=False)


@app.route("/upload-audio", methods=["GET", "POST"])
def upload_audio():
    result = ""
    inputtext = ""
    if request.method == "POST":
        if request.files:
            audio = request.files["audio"]
            audio_filename = secure_filename(audio.filename) + "_audio" + ".wav"
            audio.save(os.path.join(app.config["UPLOAD_FOLDER"], audio_filename))
            inputtext = d.process_data("./upload/" + str(audio_filename))
            return redirect("./homepage")
    return redirect("./homepage")


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    result = ""
    inputtext = ""
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image_filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_IMAGE"], image_filename))
            return redirect("./homepage")
    return redirect("./homepage")


@app.route("/qsbot", methods=["GET", "POST"])
def qsbot():
    if request.method == "POST":
        data = request.form["text_data"]
        question = request.form["question"]
        QA_input = {"question": str(question), "context": str(data)}

        res = nlp(QA_input)
        answer = res["answer"]
        with open("data.txt", "w", encoding="utf8") as f1:
            f1.write(data)
        with open("question.txt", "w", encoding="utf8") as f2:
            f2.write(question)
        with open("answer.txt", "w", encoding="utf8") as f3:
            f3.write(answer)
        return redirect("./homepage")
    return redirect("./homepage")


@app.route('/',methods = ['GET','POST'])
def index():
    result=""
    inputtext =""
    if request.method == 'GET':
        audio_file = os.listdir("./upload")
        image_file = os.listdir("./image")
        if len(audio_file) > 0 :
            inputtext = d.process_data("./upload/"+str(audio_file[0]))
            os.remove("./upload/"+str(audio_file[0]))                  
        elif len(image_file) > 0:            
            inputtext = pytesseract.image_to_string(Image.open("./image/"+str(image_file[0])), lang="vie")
            print(text)
            os.remove("./image/"+str(image_file[0]))

        return render_template('index.html',contentText = inputtext, result = result)
    if request.method == 'POST':
        # --- Function summary
        if request.form['action'] == "Summary":
            inputtext = request.form['contentText']
            result = text(str(inputtext))
        elif request.form['action'] == "Correct":
            inputtext = request.form['contentText']
            result = get_correct(inputtext)
        return render_template('index.html',contentText = inputtext, result = result)


@app.route("/homepage", methods=["GET", "POST"])
def homepage():
    result=""
    inputtext =""
    if request.method == 'GET':
        audio_file = os.listdir("./upload")
        image_file = os.listdir("./image")
        if len(audio_file) > 0 :
            inputtext = d.process_data("./upload/"+str(audio_file[0]))
            os.remove("./upload/"+str(audio_file[0]))                  
        elif len(image_file) > 0:            
            inputtext = pytesseract.image_to_string(Image.open("./image/"+str(image_file[0])), lang="vie")
            
            os.remove("./image/"+str(image_file[0]))

        return render_template('index.html',contentText = inputtext, result = result)
    if request.method == 'POST':
        # --- Function summary
        inputtext = request.form['contentText']
        result = text(str(inputtext))

        # --- Function get correct
        # inputtext = request.form['contentText']
        # result = get_correct(inputtext)
        return render_template('index.html',contentText = inputtext, result = result)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        useraccount = Users.query.filter_by(Email=request.form["username"]).first()
        if useraccount != None:
            password = useraccount.Password
            if str(request.form["password"]) != str(password):
                error = "Sai tên đăng nhập hoặc mật khẩu !!!."
            else:
                return redirect("./homepage")
        else:
            error = "Sai tên đăng nhập hoặc mật khẩu !!!."
    return render_template("login.html")


@app.route("/adminDashBoard")
def adminmanager():
    return render_template("adminDashBoard.html")


@app.route("/admin", methods=["GET", "POST"])
def admin():
    error = None

    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "admin":
            return redirect("./adminDashBoard")
    return render_template("admin.html")


@app.route("/accounts")
def manageAccounts():
    # User = Users.query.order_by(Users.all())
    return render_template("manageAccount.html")


@app.route("/deleteAccount/<int:Id_User>")
def delete_account(Id_Account):
    User_to_delete = Users.query.get_or_404(Id_Account)

    try:
        db.session.delete(User_to_delete)
        db.session.commit()
        return redirect("/manageAccounts")
    except:
        return "There was a problem deleting that task"


@app.route("/blog")
def blog():
    return render_template("blog.html")


@app.route("/profile")
def profile():
    if not g.user:
        return redirect(url_for("login"))

    return render_template("profile.html")


@app.route("/register", methods=["POST", "GET"])
def register():
    error = None
    if request.method == "POST":
        useraccountcheck = Users.query.filter_by(Email=request.form["email"]).first()
        username = request.form["username"]
        password = request.form["password"]
        email = request.form["email"]
        repassword = request.form["repassword"]
        if useraccountcheck == None:
            if repassword == password:
                new_user = Users(User_Name=username, Password=password, Email=email)
                db.session.add(new_user)
                db.session.commit()
                return redirect("/login")
            else:
                error = "Mật khẩu không trùng khớp!!!"
        else:
            error = "User account exist!"
    return render_template("register.html", error=error)


if __name__ == "__main__":
    app.run(host="localhost", port="5000", debug=True)
