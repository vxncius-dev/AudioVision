#!/usr/bin/env python3
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from os import path, makedirs, remove, listdir
from uuid import uuid4
import speech_recognition as sr
from pydub import AudioSegment
from base64 import b64encode
from io import BytesIO
import matplotlib.pyplot as plt
from librosa import load
import librosa.display
import google.generativeai as genai
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = "uploads/"
if not path.exists(app.config["UPLOAD_FOLDER"]):
    makedirs(app.config["UPLOAD_FOLDER"])

gemini_api_key = ""
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "Nenhum arquivo foi enviado."}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Nenhum arquivo selecionado."}), 400

    filename = secure_filename(file.filename)
    audio_path = path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(audio_path)
    except Exception as e:
        return jsonify({"error": "Erro ao salvar o arquivo.", "details": str(e)}), 500

    try:
        if filename.endswith(".ogg"):
            audio = AudioSegment.from_ogg(audio_path)
            wav_path = audio_path.replace(".ogg", ".wav")
            audio.export(wav_path, format="wav")
            if not path.exists(wav_path):
                return jsonify({"error": "Erro na conversão do arquivo."}), 500
        else:
            wav_path = audio_path

        transcription = transcribe_audio(wav_path)
        plot_base64 = generate_waveform(wav_path)
        unique_id = str(uuid4())

        if path.exists(audio_path):
            remove(audio_path)
        if filename.endswith(".ogg") and path.exists(wav_path):
            remove(wav_path)

        response_data = {
            "transcription": transcription,
            "plot": plot_base64,
            "id": unique_id,
        }

        clear_uploads()
        return jsonify(response_data)

    except Exception as e:
        return (
            jsonify({"error": "Erro ao processar o arquivo.", "details": str(e)}),
            500,
        )


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    transcription = data.get("transcription", "")
    if not transcription:
        return jsonify({"error": "Texto de transcrição ausente."}), 400

    analysis_result = analyze_transcription(transcription)
    return jsonify({"analysis": analysis_result})


def clear_uploads():
    for filename in listdir(app.config["UPLOAD_FOLDER"]):
        file_path = path.join(app.config["UPLOAD_FOLDER"], filename)
        try:
            if path.isfile(file_path):
                remove(file_path)
        except Exception as e:
            print(f"Erro ao tentar remover o arquivo {file_path}: {e}")


def transcribe_audio(wav_path: str) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data, language="pt-BR")
        except sr.UnknownValueError:
            return "Não foi possível entender o áudio."
        except sr.RequestError as e:
            return f"Erro ao conectar ao serviço de reconhecimento de fala: {e}"


def analyze_transcription(transcription: str) -> str:
    prompt = (
        f"Você é um assistente inteligente. O texto a seguir é uma transcrição de áudio "
        f'que pode conter erros. Tente entender e resumir o conteúdo: "{transcription}"'
    )
    try:
        response = model.generate_content(prompt)
        if response and response.candidates:
            candidate = response.candidates[0]
            gemini_response = "".join(part.text for part in candidate.content.parts)
            return gemini_response
        else:
            return "Desculpe, não consegui analisar a transcrição."
    except Exception as e:
        return f"Erro ao tentar analisar a transcrição: {e}"


def generate_waveform(audio_path: str) -> str:
    y, sr = load(audio_path, sr=None)
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, color="white")
    plt.axis("off")
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(
        buf, format="png", bbox_inches="tight", pad_inches=0.1, transparent=True
    )
    plt.close()
    buf.seek(0)
    return f"data:image/png;base64,{b64encode(buf.read()).decode('utf-8')}"


if __name__ == "__main__":
    app.run(debug=True, port=5001)
