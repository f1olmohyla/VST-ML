import os
from pedalboard import Pedalboard, Reverb, load_plugin
from pedalboard.io import AudioFile
from mido import Message # not part of Pedalboard, but convenient!

# os.chdir(r"/Users/f1ol/workspace/ukma/ML/VST-ML")

# Get the current working directory
# current_directory = os.getcwd()

# List all files and directories in the current directory
# files_and_directories = os.listdir(current_directory)

# Print the contents
# for item in files_and_directories:
#     print(item)

# Load a VST3 or Audio Unit plugin from a known path on disk:
# instrument = load_plugin("./Relica.vst3")
effect = load_plugin("./RoughRider3.vst3")

print(effect.parameters.keys())
# dict_keys([
#   'sc_hpf_hz', 'input_lvl_db', 'sensitivity_db',
#   'ratio', 'attack_ms', 'release_ms', 'makeup_db',
#   'mix', 'output_lvl_db', 'sc_active',
#   'full_bandwidth', 'bypass', 'program',
# ])

# Set the "ratio" parameter to 15
effect.ratio = 15

# Render some audio by passing MIDI to an instrument:
sample_rate = 44100

audio = instrument(
  [Message("note_on", note=60), Message("note_off", note=60, time=4)],
  duration=4, # seconds
  sample_rate=sample_rate,
)

#edit csv file

# Apply effects to this audio:
effected = effect(audio, sample_rate)

# ...or put the effect into a chain with other plugins:
board = Pedalboard([effect, Reverb()])
# ...and run that pedalboard with the same VST instance!
effected = board(audio, sample_rate)