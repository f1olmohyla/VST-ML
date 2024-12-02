import os
import numpy as np
import librosa
from pedalboard import load_plugin
from pedalboard.io import AudioFile
from mido import Message
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Paths
TEST_WAV_FILE = "test_audio.wav"
MODEL_H5 = "vst_model.h5"
MODEL_KERAS = "vst_model.keras"

# Ensure the model is in .keras format
if not os.path.exists(MODEL_KERAS):
    print("Converting .h5 model to .keras format...")
    model = load_model(MODEL_H5, custom_objects={"mse": MeanSquaredError()})
    model.save(MODEL_KERAS)
    print(f"Model successfully converted and saved as {MODEL_KERAS}.")
else:
    print("Using existing .keras model.")

# Load the model
model = load_model(MODEL_KERAS)
print("Model loaded successfully.")

# Step 1: Generate a test WAV file
def generate_test_wav(output_path, params, midi_message):
    # Load VST3 plugin
    instrument = load_plugin("VSTs/Relica.vst3")

    # Set parameters to the instrument
    for key, value in params.items():
        setattr(instrument, key, value)

    # Generate audio
    audio = instrument(
        midi_message,
        duration=3,  # seconds
        sample_rate=44100
    )

    # Save the audio to a file
    with AudioFile(output_path, "w", samplerate=44100, num_channels=audio.shape[0]) as f:
        f.write(audio)
    print(f"Test WAV file generated at {output_path}")

# Define constant and dynamic parameters
constant_params = {
    "attack": 0.0,
    "decay": 2.0,
    "sustain": 100.0,
    "release": 1.0,
    "monophonic": False,
    "disable_velocity": False,
    "volume": -8.0,
    "bypass": False
}

dynamic_params = {
    "bitcrusher": 50.0,  # Example value
    "vibrato_intensity": 20.0,  # Example value
    "vibrato_frequency": 8.0,  # Example value
    "pulse_width": 50.0,  # Example value
    "oscillator_shape_sine": True,
    "oscillator_shape_triangle": False,
    "oscillator_shape_square": False,
    "oscillator_shape_sawtooth": False,
    "lfo_shape_sine": True,
    "lfo_shape_triangle": False,
    "lfo_shape_square": False,
    "lfo_shape_sawtooth": False
}

# Combine parameters
params = {**constant_params, **dynamic_params}

# Define MIDI message
midi_message = [
    Message("note_on", note=60),
    Message("note_off", note=60, time=2),  # 2 Seconds note hold duration
]

# Generate test WAV file
generate_test_wav(TEST_WAV_FILE, params, midi_message)

# Step 2: Convert the WAV file to a spectrogram
def generate_spectrogram(wav_file):
    y, sr = librosa.load(wav_file, sr=None)

    # Compute Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalize spectrogram
    S_db = np.where(S_db == 0, 1e-6, S_db)  # Replace zeros with a small value
    S_db = S_db / np.max(S_db)

    return S_db

spectrogram = generate_spectrogram(TEST_WAV_FILE)
spectrogram = spectrogram[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

# Step 3: Predict parameters using the trained model
predicted_params = model.predict(spectrogram)
print("Predicted Parameters:", predicted_params)

# Step 4: Compare predicted and actual parameters
print("\nOriginal Parameters:")
for key, value in dynamic_params.items():
    print(f"{key}: {value}")

print("\nPredicted Parameters:")
for i, key in enumerate(dynamic_params.keys()):
    # Convert binary outputs back to True/False
    if isinstance(dynamic_params[key], bool):
        prediction = predicted_params[0][i] > 0.5  # Threshold for binary values
    else:
        prediction = predicted_params[0][i]
    print(f"{key}: {prediction}")

# Optional: Calculate MAE
actual_values = np.array([int(val) if isinstance(val, bool) else val for val in dynamic_params.values()])
predicted_values = np.array([predicted_params[0][i] for i in range(len(dynamic_params))])
mae = np.mean(np.abs(actual_values - predicted_values))
print(f"\nMean Absolute Error (MAE): {mae}")