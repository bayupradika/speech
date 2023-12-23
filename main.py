import threading
import uuid
import base64
import os
import librosa
import librosa.display
import numpy as np
import speech_recognition as sr
from flask import Flask, render_template, request, jsonify, send_file
import webbrowser
import secrets
from pydub import AudioSegment
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

browser_opened_event = threading.Event()

# Fungsi untuk memeriksa apakah pembicara sah
def is_valid_speaker(speaker_representation):
    threshold = 0.5
    return speaker_representation[0] > threshold

def read_dataset_from_txt():
    word_to_image_mapping = {}
    with open('dataset.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                word_to_image_mapping[parts[0]] = parts[1]
    return word_to_image_mapping



def high_pass_filter(audio_segment, cutoff_frequency):
    # Melakukan high-pass filter dengan memotong frekuensi rendah
    return audio_segment.high_pass_filter(cutoff_frequency)


def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Silakan mulai berbicara...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=None)
    try:
        # Mengonversi audio ke dalam format PyDub
        audio_segment = AudioSegment.from_wav(audio.frame_data, sample_width=audio.sample_width,
                                              frame_rate=audio.sample_rate, channels=audio.channel_count)
        # Melakukan high-pass filter dengan cutoff frequency 1000 Hz (misalnya)
        filtered_audio = high_pass_filter(audio_segment, 1000)
        # Mengonversi kembali ke audio PyAudio
        filtered_audio_data = filtered_audio.raw_data
        filtered_audio = sr.AudioData(filtered_audio_data, sample_rate=audio.sample_rate,
                                      sample_width=audio.sample_width, channels=audio.channel_count)
        # Mengenali teks dari audio yang sudah difilter
        text = recognizer.recognize_google(filtered_audio)
        print("Teks hasil konversi: " + text)
        # Ekstraksi fitur MFCC dan pitch
        mfcc_features = extract_mfcc_from_audio(filtered_audio)
        pitch_features = extract_pitch_from_audio(filtered_audio)
        # Membandingkan dengan data suara yang sudah ada
        matching_folder = compare_with_existing_data_speech_to_text(mfcc_features, pitch_features)
        # Hapus hasil ekstraksi jika sudah ada yang sama
        if matching_folder:
            print("Hasil ekstraksi yang sama ditemukan. Menghapus hasil ekstraksi sementara...")
            os.remove(mfcc_features)
            os.remove(pitch_features)
        return text
    except sr.UnknownValueError:
        print("Maaf, tidak dapat mengenali suara.")
        return None
    except sr.RequestError as e:
        print(f"Terjadi kesalahan pada server Google: {e}")
        return None


def extract_pitch_from_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    result = librosa.pyin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
    pitches = result[0]  # atau indeks yang sesuai untuk pitches
    magnitudes = result[1]  # atau indeks yang sesuai untuk magnitudes

    pitch = pitches[magnitudes.argmax()]
    return pitch

def extract_mfcc_from_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(sr*0.010), n_fft=int(sr*0.025))
    # Transpose MFCC matrix to match the CNN input shape (time, features)
    mfccs = np.transpose(mfccs)
    return mfccs


def save_mfcc_to_file(mfcc_values, audio_file):
    # Tentukan nama file untuk menyimpan MFCC
    mfcc_file = audio_file.replace(".wav", "_mfcc.npy")

    # Simpan nilai MFCC ke file numpy
    np.save(mfcc_file, mfcc_values)

def extract_pitch_and_mfcc_from_folder(audio_folder):
    all_pitch_values = []

    # Iterasi melalui setiap file di folder
    for filename in os.listdir(audio_folder):
        if filename.endswith(".wav"):
            audio_file_path = os.path.join(audio_folder, filename)

            # Memanggil fungsi ekstraksi pitch untuk setiap file
            pitch_value = extract_pitch_from_audio(audio_file_path)

            # Menyimpan hasil pitch
            all_pitch_values.append((audio_file_path, pitch_value))

            # Memanggil fungsi ekstraksi MFCC untuk setiap file
            mfcc_values = extract_mfcc_from_audio(audio_file_path)

            # Menyimpan hasil MFCC
            save_mfcc_to_file(mfcc_values, audio_file_path)

            # Menghapus file audio yang sudah diolah
            os.remove(audio_file_path)

    return all_pitch_values

# Contoh penggunaan untuk folder 'ayam'
ayam_folder_path = 'recordings/ayam'
ayam_pitch_values = extract_pitch_and_mfcc_from_folder(ayam_folder_path)

# Contoh penggunaan untuk folder 'anjing'
anjing_folder_path = 'recordings/anjing'
anjing_pitch_values = extract_pitch_and_mfcc_from_folder(anjing_folder_path)

# Contoh penggunaan untuk folder 'kucing'
kucing_folder_path = 'recordings/kucing'
kucing_pitch_values = extract_pitch_and_mfcc_from_folder(kucing_folder_path)

def save_recorded_audio(audio_data):
    os.makedirs('recordings', exist_ok=True)
    filename = f"recordings/index_{uuid.uuid4()}.wav"
    with open(filename, 'wb') as file:
        file.write(base64.b64decode(audio_data))

    audio = AudioSegment.from_wav(filename)

    # Proses filtering (contoh: high-pass filter)
    filtered_audio = audio.high_pass_filter(1000)

    # Normalisasi amplitudo
    normalized_audio = filtered_audio.normalize()

    # Ekstraksi fitur MFCC dan pitch
    mfcc_features = extract_mfcc_from_audio(filename)
    pitch_features = extract_pitch_from_audio(filename)

    # Membandingkan dengan data suara yang sudah ada
    matching_folder = compare_with_existing_data_speech_to_text(mfcc_features, pitch_features)

    return filename, matching_folder



def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def prepare_data():
    recordings_path = 'recordings'
    subfolders = os.listdir(recordings_path)
    files, labels = [], []
    for subfolder in subfolders:
        subfolder_path = os.path.join(recordings_path, subfolder)
        subfolder_files = os.listdir(subfolder_path)
        subfolder_files = [os.path.join(subfolder, file) for file in subfolder_files]
        files.extend(subfolder_files)
        labels.extend([subfolder] * len(subfolder_files))
    X, y = [], []
    for file, label in zip(files, labels):
        filepath = os.path.join(recordings_path, file)
        mfccs = extract_mfcc_from_audio(filepath)
        X.append(mfccs)
        y.append(label)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_array = np.array(X)
    y_array = tf.keras.utils.to_categorical(y_encoded)
    X_train, X_val, y_train, y_val = train_test_split(X_array, y_array, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val


def compare_with_existing_data_speech_to_text(mfcc_features, pitch_features):
    # Mendapatkan daftar file rekaman yang sudah ada
    existing_recordings_path = 'recordings'
    subfolders = os.listdir(existing_recordings_path)

    for subfolder in subfolders:
        subfolder_path = os.path.join(existing_recordings_path, subfolder)
        subfolder_files = os.listdir(subfolder_path)

        # Ambil salah satu file rekaman sebagai contoh
        if subfolder_files:
            example_file = os.path.join(subfolder_path, subfolder_files[0])

            # Membandingkan MFCC dan pitch
            existing_mfcc = extract_mfcc_from_audio(example_file)
            existing_pitch = extract_pitch_from_audio(example_file)

            if np.array_equal(mfcc_features, existing_mfcc) and np.array_equal(pitch_features, existing_pitch):
                # Jika kesamaan ditemukan, kembalikan label subfolder
                return subfolder
        return None


@app.route('/record_and_compare')
def record_and_compare():
    try:
        # Rekam suara dari mikrofon
        audio_data = speech_to_text()

        # Ekstraksi fitur MFCC dan pitch
        mfcc_features = extract_mfcc_from_audio(audio_data)
        pitch_features = extract_pitch_from_audio(audio_data)

        # Membandingkan dengan data suara yang sudah ada
        matching_folder = compare_with_existing_data_speech_to_text(mfcc_features, pitch_features)

        if matching_folder is not None:
            # Jika ada kesamaan, set variabel pertama di dataset
            word_to_image_mapping = read_dataset_from_txt()
            word_to_image_mapping['first_variable'] = matching_folder

            # Tentukan nama file gambar berdasarkan hasil perbandingan
            if matching_folder == 'ayam':
                image_filename = 'ayam.jpg'
            elif matching_folder == 'anjing':
                image_filename = 'anjing.jpg'
            elif matching_folder == 'kucing':
                image_filename = 'kucing.jpg'
            else:
                # Jika tidak sesuai, gunakan default.jpg
                image_filename = 'default.jpg'

            word_to_image_mapping['image_filename'] = image_filename

        return jsonify({'success': True, 'matching_folder': matching_folder, 'image_filename': image_filename})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def save_training_recording(audio_data, speaker_label, subfolder):
    os.makedirs(os.path.join('recordings', subfolder), exist_ok=True)
    filename = f"recordings/{subfolder}/{speaker_label}_{uuid.uuid4()}.wav"

    with open(filename, 'wb') as file:
        file.write(base64.b64decode(audio_data))

    audio = AudioSegment.from_wav(filename)

    filtered_audio = audio.high_pass_filter(1000)

    normalized_audio = filtered_audio.normalize()

    mfcc_features = extract_mfcc_from_audio(filename)
    pitch_features = extract_pitch_from_audio(filename)
    return filename

@app.route('/save_training_recording', methods=['POST'])
def save_training_recording_endpoint():
    try:
        audio_data = request.json.get('audio_data')
        speaker_label = request.json.get('speaker_label')

        # Menentukan subfolder berdasarkan speaker_label
        if speaker_label == 'ayam':
            subfolder = 'ayam'
        elif speaker_label == 'anjing':
            subfolder = 'anjing'
        elif speaker_label == 'kucing':
            subfolder = 'kucing'
        else:
            # Jika label tidak cocok dengan yang diharapkan, simpan dalam subfolder default
            subfolder = 'default'

        # Menyimpan rekaman dalam subfolder yang sesuai
        filename = save_training_recording(audio_data, speaker_label, subfolder)

        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Route untuk halaman pelatihan suara
@app.route('/train')
def train_page():
    subfolders = [subfolder for subfolder in os.listdir('recordings') if os.path.isdir(os.path.join('recordings', subfolder))]
    return render_template('train.html', subfolders=subfolders)


@app.route('/')
def index():
    word_to_image_mapping = read_dataset_from_txt()
    return render_template('index.html', dataset=word_to_image_mapping)


def open_browser():
    global browser_opened_event

    # Check if the server is already running
    if not browser_opened_event.is_set():
        webbrowser.open('http://127.0.0.1:5001')
        browser_opened_event.set()

if __name__ == '__main__':
    t = threading.Thread(target=open_browser)
    t.start()
    app.run(debug=True, threaded=True, host='127.0.0.1', port=5001, use_reloader=False)

    # Persiapkan data latih dan validasi
    X_train, X_val, y_train, y_val = prepare_data()
