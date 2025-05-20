from flask import Flask, request, jsonify, render_template
import build
from query import getAnswer
from flask_cors import CORS

app = Flask(__name__, static_folder="public", static_url_path="/")
CORS(app)


@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/store', methods=["POST"])
def store():
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nama file tidak valid"}), 400
    
    file_content = file.read().decode("utf-8")
    docs = [line.strip() for line in file_content.splitlines() if line.strip()]
    build.runBuild(docs)
    return jsonify({"message": "File berhasil diunggah dan dibaca"})


@app.route('/search', methods=['POST'])
def search_query():
    try :
        body = request.get_json()
        jawaban = getAnswer(body["question"])
        return jawaban
    except FileNotFoundError as e:
        return e, 404
    except Exception:
        return "internal server error", 500


if __name__ == '__main__':
    app.run(debug=True)
