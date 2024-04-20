from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import requests
import json
from collections import namedtuple
from urllib.parse import urlencode

# set this from your main program
# import faster_whisper_remote_proxy 
# faster_whisper_remote_proxy.remote_url = "http://..."
remote_url: str = ""

def convertToNamedTuple(name, dictionary):
    return namedtuple( name, dictionary.keys())(**dictionary)

# instead "from faster_whisper.vad import VadOptions" to prevent installing all dependencies
class VadOptions(NamedTuple):
    """VAD options.

    Attributes:
      threshold: Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
        probabilities ABOVE this value are considered as SPEECH. It is better to tune this
        parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
      min_speech_duration_ms: Final speech chunks shorter min_speech_duration_ms are thrown out.
      max_speech_duration_s: Maximum duration of speech chunks in seconds. Chunks longer
        than max_speech_duration_s will be split at the timestamp of the last silence that
        lasts more than 100ms (if any), to prevent aggressive cutting. Otherwise, they will be
        split aggressively just before max_speech_duration_s.
      min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms
        before separating it
      window_size_samples: Audio chunks of window_size_samples size are fed to the silero VAD model.
        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate.
        Values other than these may affect model performance!!
      speech_pad_ms: Final speech chunks are padded by speech_pad_ms each side
    """

    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    window_size_samples: int = 1024
    speech_pad_ms: int = 400

# instead "from faster_whisper.transcribe import Segment, TranscriptionInfo" to prevent installing all dependencies
class Word(NamedTuple):
    start: float
    end: float
    word: str
    probability: float

class Segment(NamedTuple):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Word]]


class TranscriptionOptions(NamedTuple):
    beam_size: int
    best_of: int
    patience: float
    length_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    log_prob_threshold: Optional[float]
    no_speech_threshold: Optional[float]
    compression_ratio_threshold: Optional[float]
    condition_on_previous_text: bool
    prompt_reset_on_temperature: float
    temperatures: List[float]
    initial_prompt: Optional[Union[str, Iterable[int]]]
    prefix: Optional[str]
    suppress_blank: bool
    suppress_tokens: Optional[List[int]]
    without_timestamps: bool
    max_initial_timestamp: float
    word_timestamps: bool
    prepend_punctuations: str
    append_punctuations: str
    max_new_tokens: Optional[int]
    clip_timestamps: Union[str, List[float]]
    hallucination_silence_threshold: Optional[float]


class TranscriptionInfo(NamedTuple):
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: Optional[List[Tuple[str, float]]]
    transcription_options: TranscriptionOptions
    vad_options: VadOptions


class WhisperModelRemoteProxy :
    def __init__(
        self,
        model_size_or_path: str,
        device: str = "auto",
        device_index: Union[int, List[int]] = 0,
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
    ):
        pass

    @property
    def supported_languages(self) -> List[str]:
        """The languages supported by the model."""
        return ["en"]

    def _get_feature_kwargs(self, model_path) -> dict:
        return {}

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        temperature: Union[float, List[float], Tuple[float, ...]] = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        compression_ratio_threshold: Optional[float] = 2.4,
        log_prob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        prompt_reset_on_temperature: float = 0.5,
        initial_prompt: Optional[Union[str, Iterable[int]]] = None,
        suppress_blank: bool = True,
        suppress_tokens: Optional[List[int]] = [-1],
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        vad_filter: bool = False,
        vad_parameters: Optional[Union[dict, VadOptions]] = None,
        hallucination_silence_threshold: Optional[float] = None,
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        
        def segment_generator(self, lines):
            # Iterate over the response content as it arrives
             for line in lines:
                # Decode JSON string to dictionary
                data = json.loads(line)

                # Process depending on returned information
                if "Segment" in data:  # Process segment
                    # Yield the processed segment
                    yield convertToNamedTuple('Segment', data["Segment"])
    
        parameters = {
            'language': language,
            'task': task,
            'beam_size': beam_size,
            'best_of': best_of,
            'patience': patience,
            'length_penalty': length_penalty,
            'repetition_penalty': repetition_penalty,
            'no_repeat_ngram_size': no_repeat_ngram_size,
            # 'temperature': temperature, special handling see below
            'compression_ratio_threshold': compression_ratio_threshold,
            'log_prob_threshold': log_prob_threshold,
            'no_speech_threshold': no_speech_threshold,
            'condition_on_previous_text': condition_on_previous_text,
            'prompt_reset_on_temperature': prompt_reset_on_temperature,
            'initial_prompt': initial_prompt,
            'suppress_blank': suppress_blank,
            'suppress_tokens': suppress_tokens,
            'word_timestamps': word_timestamps,
            'prepend_punctuations': prepend_punctuations,
            'append_punctuations': append_punctuations,
            'hallucination_silence_threshold': hallucination_silence_threshold,
            'vad_filter': vad_filter,
            'vad_parameters': json.dumps(vad_parameters) if vad_parameters else None,
        }

        # temperature needs special handling because can:
        # temperature: Union[float, List[float], Tuple[float, ...]]
        if temperature:
            if type(temperature) is float:
                parameters['temperature'] = [temperature] # create a list
            parameters['temperature'] = json.dumps(temperature)

        # remove empty/None paramters
        parameters = {k: v for k, v in parameters.items() if v is not None}
        try:
            # Construct the query string
            query_string = urlencode( parameters)
            if query_string is not None:
                query_string = '?' + query_string

            if isinstance(audio, str):
                files = {"file": open(audio, "rb")}
                # send request
                r = requests.post( remote_url + query_string, files=files, stream=True)

            elif isinstance(audio, np.ndarray):
                # Case when audio is a NumPy array
                audio_data = audio.flatten().astype("float32")
                r = requests.post( remote_url + query_string, data=audio_data.tobytes(), stream=True)

            # Iterate over the response content as it arrives
            lines = r.iter_lines()
            first_line = next(lines)
            data = json.loads(first_line)

            # first entry should be info
            if "TranscriptionInfo" in data: # process info
                info = convertToNamedTuple('TranscriptionInfo', data["TranscriptionInfo"])

            return segment_generator(self, lines), info
                   
        except requests.exceptions.RequestException as e:
            # Handle any request exceptions, such as a connection error or timeout
            print(f"Error connecting to the remote URL:{remote_url}", e)
            return None