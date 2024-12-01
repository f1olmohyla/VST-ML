import os
from pedalboard import Pedalboard, Reverb, load_plugin
from pedalboard.io import AudioFile
from mido import Message # not part of Pedalboard, but convenient!
from utils import create_new_folder
import pandas as pd


# Load a VST3 or Audio Unit plugin from a known path on disk:
instrument = load_plugin("VSTs/Relica.vst3")

# Print param ranges
parameter_info = {}
parameters = instrument.parameters
for param_name, param in parameters.items():
    min_val = param.min_value
    max_val = param.max_value
    parameter_info[param_name] = {
        "param_name": param_name,
        "min": min_val,
        "max": max_val,
    }
    print(f"{param_name}:")
    print(f"    Range: [{min_val}, {max_val}]")

# Preset constant parameters
constant_params_map = {
    "attack": 0.0,
    "decay": 2.0,
    "sustain": 100.0,
    "release": 1.0,
    "monophonic": False,
    "disable_velocity": False,
    "volume": -8.0,
    "bypass": False
}

dynamic_parameters = [
    {
        "param_name": param_name,
        "min": info["min"],
        "max": info["max"],
    }
    for param_name, info in parameter_info.items()
    if param_name not in constant_params_map
]

# Print dynamic parameters
print("\nDynamic Parameters:")
for param in dynamic_parameters:
    print(param)

# Create output folder
audio_output_folder = "audio_files"
create_new_folder(audio_output_folder)

# Define helper function to create sequential oscillator or LFO shapes
def generate_shape_combinations(param_names):
    combinations = []
    for i, name in enumerate(param_names):
        combination = {key: False for key in param_names}
        combination[name] = True
        combinations.append(combination)
    return combinations

# Generate oscillator shape combinations
oscillator_shapes = [
    'oscillator_shape_sine',
    'oscillator_shape_triangle',
    'oscillator_shape_square',
    'oscillator_shape_sawtooth',
]
oscillator_combinations = generate_shape_combinations(oscillator_shapes)

# Generate LFO shape combinations
lfo_shapes = [
    'lfo_shape_sine',
    'lfo_shape_triangle',
    'lfo_shape_square',
    'lfo_shape_sawtooth',
]
lfo_combinations = generate_shape_combinations(lfo_shapes)

# Generate step-based ranges for continuous parameters
def generate_values(param):
    if isinstance(param['min'], bool):  # Skip boolean parameters
        return []
    range_size = param['max'] - param['min']
    step = range_size / 4  # Step is 1/5th of the range
    return [round(param['min'] + i * step, 1) for i in range(5)]  # Include both ends

bitcrusher_values = generate_values(next(p for p in dynamic_parameters if p['param_name'] == 'bitcrusher'))
vibrato_intensity_values = generate_values(next(p for p in dynamic_parameters if p['param_name'] == 'vibrato_intensity'))
vibrato_frequency_values = generate_values(next(p for p in dynamic_parameters if p['param_name'] == 'vibrato_frequency'))
pulse_width_values = generate_values(next(p for p in dynamic_parameters if p['param_name'] == 'pulse_width'))

# Combine all parameters into a list of maps
result = []
file_counter = 0
for bitcrusher in bitcrusher_values:
    for vibrato_intensity in vibrato_intensity_values:
        for vibrato_frequency in vibrato_frequency_values:
            for pulse_width in pulse_width_values:
                for oscillator_shape in oscillator_combinations:
                    for lfo_shape in lfo_combinations:
                        param_map = {
                            'file_name': str(file_counter) + ".wav",
                            'bitcrusher': bitcrusher,
                            'vibrato_intensity': vibrato_intensity,
                            'vibrato_frequency': vibrato_frequency,
                            'pulse_width': pulse_width,
                            **oscillator_shape,
                            **lfo_shape,
                            **constant_params_map,
                        }
                        file_counter += 1
                        result.append(param_map)

# Output the result
print(f"Generated {len(result)} parameter maps.")

# Create a DataFrame to organize data
df = pd.DataFrame(result)

# Save the DataFrame to a CSV file
output_file_path = "vst_plugin_parameters.csv"
df.to_csv(output_file_path, index=False)

print(f"CSV file saved at {output_file_path}")

# Render some audio by passing MIDI to an instrument:
sample_rate = 44100
midi_message = [
    Message("note_on", note=60),
    Message("note_off", note=60, time=2), # 2 Seconds note hold duration
]

# Set constant parameters to instrument
# Iterate through general_parameters and generate audio files
for idx, param_map in enumerate(result):
    # Set all parameters to the instrument
    for key, value in param_map.items():
        setattr(instrument, key, value)

    file_name = param_map["file_name"]

    # Render audio using the current parameter settings
    audio = instrument(
        midi_message,
        duration=3,  # seconds
        sample_rate=sample_rate,
    )

    # Save the audio to a file
    output_file = os.path.join(audio_output_folder, f'{param_map["file_name"]}')
    with AudioFile(output_file, "w", samplerate=sample_rate, num_channels=audio.shape[0]) as f:
        f.write(audio)

    print(f"Generated file: {output_file}")