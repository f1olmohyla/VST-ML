import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Paths
AUDIO_DIR = "audio_files"  # Folder with WAV files
CSV_FILE = "vst_plugin_parameters.csv"  # CSV file with annotations
SPECTROGRAM_DIR = "spectrograms"  # Directory for preprocessed spectrograms
MODEL_FILE = "vst_model.h5"  # File to save the trained model

# Create the spectrogram directory if it doesn't exist
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)

# Step 1: Generate spectrograms from WAV files
def generate_spectrograms():
    print("Generating spectrograms...")
    for wav_file in os.listdir(AUDIO_DIR):
        if wav_file.endswith(".wav"):
            # Load audio
            audio_path = os.path.join(AUDIO_DIR, wav_file)
            y, sr = librosa.load(audio_path, sr=None)

            # Compute Mel Spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)

            # Save spectrogram as NumPy array
            spectrogram_path = os.path.join(SPECTROGRAM_DIR, wav_file.replace(".wav", ".npy"))
            np.save(spectrogram_path, S_db)

    print("Spectrogram generation complete!")

# Step 2: Load spectrograms and annotations
def load_data(csv_file, spectrogram_dir):
    print("Loading data...")
    df = pd.read_csv(csv_file)

    # Ensure all non-numeric data is converted to numeric
    for col in df.columns[1:]:  # Skip "file_name" column
        if df[col].dtype == "bool":  # Convert booleans to integers
            df[col] = df[col].astype(int)
        elif df[col].dtype == "object":  # Check for non-numeric data
            raise ValueError(f"Column '{col}' contains non-numeric data. Please preprocess it.")

    X = []
    y = []

    for _, row in df.iterrows():
        # Load spectrogram
        spectrogram_path = os.path.join(spectrogram_dir, row["file_name"].replace(".wav", ".npy"))
        spectrogram = np.load(spectrogram_path)

        # Ensure all target labels are numeric
        y_values = row.drop("file_name").values
        y_values = np.array([float(val) if isinstance(val, (int, float)) else val for val in y_values])

        X.append(spectrogram)
        y.append(y_values)

    # Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y, dtype=np.float32)  # Ensure y is float32

    print(f"Loaded {len(X)} samples with shape {X[0].shape} and {len(y[0])} parameters.")
    return X, y

# Step 3: Build and train the model
def train_model(X, y):
    # Normalize spectrograms
    X = np.where(X == 0, 1e-6, X)  # Replace zeros with a small value
    X = X / np.max(X)  # Normalize

    # Add channel dimension for CNNs
    X = X[..., np.newaxis]  # Shape: (num_samples, freq_bins, time_bins, 1)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")

    # Build the model
    model = models.Sequential([
        Input(shape=(128, X.shape[2], 1)),
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),  # Dropout for regularization
        layers.Dense(y_train.shape[1])  # Output layer
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # Increased epochs
        batch_size=32,
        callbacks=[early_stopping, lr_scheduler]
    )

    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    # Save the trained model
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)  # Delete existing model
    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    return model, history

# Main workflow
if __name__ == "__main__":
    # generate_spectrograms()

    # Load dataset
    X, y = load_data(CSV_FILE, SPECTROGRAM_DIR)

    # Train the model
    train_model(X, y)
