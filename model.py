import os 
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def extract_features(audio_path):
    try:
        audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13), axis=1)
        return mfccs
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

emotions = ['angry', 'disgust', 'Fear', 'happy', 'neutral', 'Pleasant_surprise', 'Sad']
X = []
y = []

base_path = r'C:\Users\ganes\OneDrive - K L University\Desktop\Emotion_Speech_Recognition\Dataset'

for emotion in emotions:
    audio_files = [os.path.join(base_path, emotion, file) for file in os.listdir(os.path.join(base_path, emotion))]
    for audio_file in audio_files:
        features = extract_features(audio_file)
        if features is not None:
            X.append(features)
            y.append(emotions.index(emotion))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
