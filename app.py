import os
from flask import Flask, request, render_template
from model import model, extract_features, emotions
from pydub import AudioSegment
import mysql.connector

app = Flask(__name__)

 
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Devalla@1234',
    'database': 'emotion'
}

@app.route("/", methods=["GET", "POST"])
def upload_audio():
    if request.method == "POST":
        audio_file = request.files["file"]
        if audio_file:
            audio_file_path = "temp_audio.wav"
            audio_file.save(audio_file_path)

            detected_emotion, audio_info = recognize_emotion_from_audio(audio_file_path)

            os.remove(audio_file_path)   

             
            save_results_to_database(detected_emotion, audio_info)

            return render_template("result.html", detected_emotion=detected_emotion, audio_info=audio_info)

    return render_template("upload.html")

def save_results_to_database(detected_emotion, audio_info):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Create a SQL query to insert the results into a table in the "emotion" schema
        query = "INSERT INTO emotion.results (emotion, sample_rate, channels, decibel, bit_depth, audio_length) VALUES (%s, %s, %s, %s, %s, %s)"
        values = (detected_emotion, audio_info['sample_rate'], audio_info['channels'], audio_info['decibel'], audio_info['bit_depth'], audio_info['audio_length'])
        cursor.execute(query, values)

        conn.commit()
    except Exception as e:
        print(f"Error saving to the database: {e}")
    finally:
        cursor.close()
        conn.close()

def recognize_emotion_from_audio(audio_file):
    features = extract_features(audio_file)

    if features is not None:
        emotion_index = model.predict([features])[0]
        predicted_emotion = emotions[emotion_index]

        
        audio = AudioSegment.from_file(audio_file)
        audio_info = {
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "decibel": audio.dBFS,
            "bit_depth": audio.sample_width * 8,
            "audio_length": len(audio) / 1000
        }

        return predicted_emotion, audio_info
    else:
        return "Error extracting features", None

if __name__ == "__main__":
    app.run(debug=True)

