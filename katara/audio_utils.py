from typing import Union, List, Tuple
import numpy as np
import soundfile as sf
import librosa
import os, time
import moonshine
from pathlib import Path
from tqdm.auto import tqdm


def moonshine_transcriber(processed_audio, model_name="moonshine/tiny"):
    stime = time.time()

    # transcribe with moonshine
    text_extract = moonshine.transcribe(processed_audio, model_name)

    time_taken = time.time() - stime
    print(f"transcribed in {time_taken:.2f}s")

    return text_extract[0]


def audiofile_crawler(
    directory: Union[str, Path] = Path.home(),
) -> Tuple[List[str], int]:
    audio_extensions = (".mp3", ".wav", ".flac", ".ogg")
    # audio_files = []

    dir_path = Path(directory)
    print(dir_path)

    # Use a generator expression to find all files with the given extensions
    audio_files = [
        str(file)
        for ext in tqdm(audio_extensions, ncols=50)
        for file in dir_path.rglob(f"*{ext}")
    ]

    matching_files = filter_files(audio_files)

    print(f"{len(matching_files)} valid files out of {len(audio_files)} files.")

    return matching_files, len(matching_files)


# read audio file into numpy array/torch tensor from file path
def read_audio(audio_file: Union[str, os.PathLike], sample_rate=22400) -> np.ndarray:
    waveform, _ = librosa.load(audio_file, sr=sample_rate)
    waveform = trimpad_audio(waveform, 30)

    return waveform


# trimming audio to a fixed length for all tasks


def trimpad_audio(audio: np.ndarray, max_duration: int, sample_rate=22400) -> np.ndarray:
    # calculate total number of samples
    samples = int(sample_rate * max_duration)

    # cut off excess samples if beyong length, or pad to req. length
    if len(audio) > samples:
        audio = audio[:samples]
    else:
        pad_width = samples - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="reflect")

    return audio


# filter corrupt files and return list of soundfile-readable files only


def filter_files(audio_files: List) -> List[str]:
    sane_files = []

    for file in audio_files:
        try:
            sf.read(file)
            sane_files.append(file)
        except:
            continue

    return sane_files
