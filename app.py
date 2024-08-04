from flask import Flask, render_template, request
from markupsafe import Markup
from dotenv import load_dotenv
import os
from finbot.retrieval_generation import generation
from finbot.data_ingestion import ingestdata

app = Flask(__name__)

load_dotenv()

vstore = ingestdata("done")
chain = generation(vstore)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    result = chain.invoke(input)
    print("Response: ", result)
    return Markup(result)

if __name__ == '__main__':
    app.run(port=8080)
