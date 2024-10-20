import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pygame
import math
from sklearn.decomposition import PCA
from LineGraph import LineGraph
from Graph2D import Graph2D
from Text import Text

from Spring import Spring


music_path_for_preview = "../music/Starfield.mp3"
separated_folder_path = "../separated/htdemucs/Starfield"

print("Loading full song")
full_y, full_sr = librosa.load(music_path_for_preview, dtype=None)

spectral_contrast = librosa.feature.spectral_contrast(y=full_y, sr=full_sr)

print("Loading drums stem")
drums_y, drums_sr = librosa.load(separated_folder_path + "/drums.mp3", dtype=None)

print("Loading vocals stem")
vocals_y, vocals_sr = librosa.load(separated_folder_path + "/vocals.mp3", dtype=None)

print("Loading bass stem")
bass_y, bass_sr = librosa.load(separated_folder_path + "/bass.mp3", dtype=None)

print("Loading other stem")
other_y, other_sr = librosa.load(separated_folder_path + "/other.mp3", dtype=None)

from scipy.signal import butter, filtfilt
def get_frequency_bands(y, sr):
    # Define frequency ranges
    KICK_CUTOFF = 150  # Hz
    SNARE_LOW = 200    # Hz
    SNARE_HIGH = 2000  # Hz

    # Create butterworth filters
    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_lowpass(cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low')
        return b, a

    def butter_highpass(cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high')
        return b, a

    # Apply filters
    # Kick (low frequencies)
    b, a = butter_lowpass(KICK_CUTOFF, sr)
    y_kick = filtfilt(b, a, y)

    # Snare (mid frequencies)
    b, a = butter_bandpass(SNARE_LOW, SNARE_HIGH, sr)
    y_snare = filtfilt(b, a, y)

    # Hi-hats/Percussion (high frequencies)
    b, a = butter_highpass(SNARE_HIGH, sr)
    y_hihat = filtfilt(b, a, y)

    # Get envelopes (using RMS energy)
    frame_length = 2048
    hop_length = 512

    kick_env = librosa.feature.rms(y=y_kick, frame_length=frame_length, hop_length=hop_length)[0]
    snare_env = librosa.feature.rms(y=y_hihat, frame_length=frame_length, hop_length=hop_length)[0]

    return kick_env, snare_env, hop_length

kick_env, snare_env, hop_length = get_frequency_bands(drums_y, drums_sr)

def kick_graph_data(kick_env, time):
  frame = librosa.time_to_frames(time, sr=drums_sr, hop_length=hop_length)

  return kick_env[frame]

def snare_graph_data(hihat_env, time):
  frame = librosa.time_to_frames(time, sr=drums_sr, hop_length=hop_length)

  return hihat_env[frame]

kick_graph = LineGraph(x=100, y=120, width=200, height=20)
snare_graph = LineGraph(x=100, y=120 + 20 + 20, width=200, height=20)

# Extract tempo and beats
tempo, beats = librosa.beat.beat_track(y=drums_y, sr=drums_sr, units="time",)
onsets = librosa.onset.onset_detect(y=drums_y, sr=drums_sr, units="time", )

bass_tonnetz = librosa.feature.tonnetz(y=bass_y, sr=bass_sr,)
other_tonnetz = librosa.feature.tonnetz(y=other_y, sr=other_sr,)


# Collapse to 2D via dimensional reduction
bass_vibe = PCA(n_components=2).fit_transform(bass_tonnetz.T)
other_vibe = PCA(n_components=2).fit_transform(other_tonnetz.T)

pygame.init()
screen = pygame.display.set_mode((1280,720))
clock = pygame.time.Clock()

running = True

text_font = pygame.font.SysFont("Input Mono", 24)

def draw_text(text, font, text_col, x, y):
  font = text_font
  img = font.render(text, True, text_col)
  screen.blit(img, (x, y))


def binary_search_prefer_left(arr, target):
    if not arr.any():
        return 0, float('inf')  # Return 0 index and infinite distance for empty array

    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid, 0  # Exact match, distance is 0

        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # Clamp the left index to be within the array bounds
    left = max(0, min(left, len(arr) - 1))

    return left, abs(arr[left] - target)

def magnitude(values):
  out = 0
  for value in values:
    value = value ** 2
    out = out + value

  out = math.sqrt(out)

  return out


def beat(beats, time):
  index, distance = binary_search_prefer_left(beats, time)

  return beats[index], distance

def onset(onsets, time):
   index, distance = binary_search_prefer_left(onsets, time)

   return onsets[index], distance

beat_tempo_spring = Spring(pygame.Vector2(0, 0), pygame.Vector2(0, 0))
snare_tempo_spring = Spring(pygame.Vector2(0, 0), pygame.Vector2(0, 0))

def beat_tempo(onsets, kick_env, time):
  beat_time, beat_distance = onset(onsets, time)
  kick_env_data = kick_graph_data(kick_env, beat_time)

  if beat_distance <= 0.03:
    beat_tempo_spring.reset(pygame.Vector2(kick_env_data * 4, 0))

  beat_tempo_spring.update(1/60)

def snare_tempo(onsets, snare_env, time):
  beat_time, beat_distance = onset(onsets, time)
  snare_env_data = snare_graph_data(snare_env, beat_time)

  if beat_distance <= 0.03:
    snare_tempo_spring.reset(pygame.Vector2(snare_env_data * 4, 0))

  snare_tempo_spring.update(1/60)

tempo_label = Text(10, 200, text="TEMPO")
transient_label = Text(10, 240, text="TRANSIENT")
beat_graph = LineGraph(x=100, y=200, width=200, height=20)
onset_graph = LineGraph(x=100, y=220 + 20, width=200, height=20)

beat_tempo_label = Text(10, 300, text="BEAT-TEMPO")
snare_tempo_label = Text(10, 320 + 20, text="SNARE-TEMPO")
beat_tempo_graph = LineGraph(x=100, y=300, width=200, height=20)
snare_tempo_graph = LineGraph(x=100, y=320 + 20, width=200, height=20)

def bass_vibe_data(bass_vibe, time):
  frame = librosa.time_to_frames(time, sr=bass_sr, hop_length=512)

  return bass_vibe[frame]

def other_vibe_data(other_vibe, time):
  frame = librosa.time_to_frames(time, sr=other_sr, hop_length=512)

  return other_vibe[frame]


pygame.mixer.init()
pygame.mixer.music.load(music_path_for_preview)
pygame.mixer.music.play()


beat_max_distance = 0
onset_max_distance = 0
iters = 0

# Initialize UI elements
title_text = Text(10, 10)
tempo_text = Text(10, 30)
transient_text = Text(10, 50)
kick_label = Text(10, 120, text="KICK")
snare_label = Text(10, 160, text="SNARE")
bass_label = Text(400, 120, text="BASS")
other_label = Text(400, 340, text="OTHER")

# Initialize graphs
kick_graph = LineGraph(x=180, y=120, width=200, height=20)
snare_graph = LineGraph(x=180, y=160, width=200, height=20)
beat_graph = LineGraph(x=180, y=200, width=200, height=20)
onset_graph = LineGraph(x=180, y=240, width=200, height=20)

bass_graph = Graph2D(x=400, y=120, width=200, height=200)
other_graph = Graph2D(x=400, y=340, width=200, height=200)


while running:


  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False

  time = pygame.mixer.music.get_pos() / 1000

  screen.fill("black")

  # Update text content
  title_text.set_text(f"{separated_folder_path} - TIME: {time:.2f}")

  last_beat_time, beat_distance = beat(beats, time)
  last_onset_time, onset_distance = onset(onsets, time)

  if iters > 30:
     beat_max_distance = 0
     onset_max_distance = 0
     iters = 0

  iters += 1

  onset_max_distance = max(onset_max_distance, onset_distance)
  beat_max_distance = max(beat_max_distance, beat_distance)

  beat_intensity = int(((beat_distance/beat_max_distance)) * 255)
  onset_intensity = int(((onset_distance/onset_max_distance)) * 255)

  tempo_text.set_text(f"TEMPO - {beat(beats, time)}")
  tempo_text.set_color(pygame.Color(beat_intensity, beat_intensity, beat_intensity))

  transient_text.set_text(f"TRANSIENT - {onset(onsets, time)}")
  transient_text.set_color(pygame.Color(onset_intensity, onset_intensity, onset_intensity))

  # Update graph data
  _beat_data = 1 if beat(beats, time)[1] <= 0.03 else 0
  _onset_data = 1 if onset(onsets, time)[1] <= 0.03 else 0

  beat_graph.add_point(_beat_data)
  onset_graph.add_point(_onset_data)

  _bass_vibe_data = bass_vibe_data(bass_vibe, time)
  _other_vibe_data = other_vibe_data(other_vibe, time)

  bass_graph.add_point(_bass_vibe_data[0], _bass_vibe_data[1])
  other_graph.add_point(_other_vibe_data[0], _other_vibe_data[1])

  kick_env_data = kick_graph_data(kick_env, time)
  snare_env_data = snare_graph_data(snare_env, time)

  beat_tempo(onsets, kick_env, time)
  snare_tempo(onsets, snare_env, time)
  beat_tempo_graph.add_point(beat_tempo_spring.get_position()[0])
  snare_tempo_graph.add_point(snare_tempo_spring.get_position()[0])

  kick_graph.add_point(kick_env_data)
  snare_graph.add_point(snare_env_data)

  # Generate color based on bass and other vibe
  hue = int((math.atan2(_bass_vibe_data[1] + _other_vibe_data[1] * 0.5, _bass_vibe_data[0] + _other_vibe_data[0] * 0.5) + math.pi) * (180 / math.pi))  # Use atan2 to map x,y to angle, then convert to 0-360 range

  brightness = min(max(pow(int(beat_tempo_spring.get_position()[0] * 100), 3), 50), 100)
  snare_brightness = min(max(pow(int(snare_tempo_spring.get_position()[0] * 100), 3), 20), 100)

  # Convert HSV to RGB
  color = pygame.Color(0)

  color.hsva = (hue, 100 - snare_brightness, brightness, 100)

  # Create a surface with the generated color
  color_surface = pygame.Surface((100, 100))
  color_surface.fill(color)

  # Draw the color surface
  screen.blit(color_surface, (50, 400))

  # Create and draw a text label for the color
  color_label = Text(50, 510, font_size=18, text="Generated Color")
  color_label.draw(screen)

  # Draw UI elements
  title_text.draw(screen)
  tempo_text.draw(screen)
  transient_text.draw(screen)
  tempo_label.draw(screen)
  transient_label.draw(screen)
  kick_label.draw(screen)
  snare_label.draw(screen)
  bass_label.draw(screen)
  other_label.draw(screen)

  beat_tempo_label.draw(screen)
  snare_tempo_label.draw(screen)

  # Draw graphs
  kick_graph.draw(screen)
  snare_graph.draw(screen)
  beat_graph.draw(screen)
  onset_graph.draw(screen)
  beat_tempo_graph.draw(screen)
  snare_tempo_graph.draw(screen)
  bass_graph.draw(screen, "white")
  other_graph.draw(screen, "white")

  pygame.display.flip()

  clock.tick(60)

pygame.quit()
