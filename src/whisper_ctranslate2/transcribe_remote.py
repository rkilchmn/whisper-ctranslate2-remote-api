from .writers import format_timestamp
from typing import NamedTuple, Optional, List, Union
import tqdm
import sys
from faster_whisper import WhisperModel
from .languages import LANGUAGES
from typing import BinaryIO
import numpy as np
import requests
import json
from .transcribe import TranscriptionOptions
from collections import namedtuple

def convertToNamedTuple(name, dictionary):
    return namedtuple( name, dictionary.keys())(**dictionary)

system_encoding = sys.getdefaultencoding()

if system_encoding != "utf-8":

    def make_safe(string):
        return string.encode(system_encoding, errors="replace").decode(system_encoding)

else:

    def make_safe(string):
        return string
    
class Transcribe_remote:
    def _get_colored_text(self, words):
        k_colors = [
            "\033[38;5;196m",
            "\033[38;5;202m",
            "\033[38;5;208m",
            "\033[38;5;214m",
            "\033[38;5;220m",
            "\033[38;5;226m",
            "\033[38;5;190m",
            "\033[38;5;154m",
            "\033[38;5;118m",
            "\033[38;5;82m",
        ]

        text_words = ""

        n_colors = len(k_colors)
        for word in words:
            p = word.probability
            col = max(0, min(n_colors - 1, (int)(pow(p, 3) * n_colors)))
            end_mark = "\033[0m"
            text_words += f"{k_colors[col]}{word.word}{end_mark}"

        return text_words

    def _get_vad_parameters_dictionary(self, options):
        vad_parameters = {}

        if options.vad_threshold:
            vad_parameters["threshold"] = options.vad_threshold

        if options.vad_min_speech_duration_ms:
            vad_parameters[
                "min_speech_duration_ms"
            ] = options.vad_min_speech_duration_ms

        if options.vad_max_speech_duration_s:
            vad_parameters["max_speech_duration_s"] = options.vad_max_speech_duration_s

        if options.vad_min_silence_duration_ms:
            vad_parameters[
                "min_silence_duration_ms"
            ] = options.vad_min_silence_duration_ms

        return vad_parameters

    def __init__(
        self,
        remote_url: str
    ):
        self.remote_url=remote_url

    def inference(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        task: str,
        language: str,
        verbose: bool,
        live: bool,
        options: TranscriptionOptions,
    ):
        # vad_parameters = self._get_vad_parameters_dictionary(options)

        # segments, info = self.model.transcribe(
        #     audio=audio,
        #     language=language,
        #     task=task,
        #     beam_size=options.beam_size,
        #     best_of=options.best_of,
        #     patience=options.patience,
        #     length_penalty=options.length_penalty,
        #     repetition_penalty=options.repetition_penalty,
        #     no_repeat_ngram_size=options.no_repeat_ngram_size,
        #     temperature=options.temperature,
        #     compression_ratio_threshold=options.compression_ratio_threshold,
        #     log_prob_threshold=options.log_prob_threshold,
        #     no_speech_threshold=options.no_speech_threshold,
        #     condition_on_previous_text=options.condition_on_previous_text,
        #     prompt_reset_on_temperature=options.prompt_reset_on_temperature,
        #     initial_prompt=options.initial_prompt,
        #     suppress_blank=options.suppress_blank,
        #     suppress_tokens=options.suppress_tokens,
        #     word_timestamps=True if options.print_colors else options.word_timestamps,
        #     prepend_punctuations=options.prepend_punctuations,
        #     append_punctuations=options.append_punctuations,
        #     hallucination_silence_threshold=options.hallucination_silence_threshold,
        #     vad_filter=options.vad_filter,
        #     vad_parameters=vad_parameters,
        # )

        try:
            files = {"audio_file": open(audio, "rb")}
            r = requests.post( self.remote_url, files=files, stream=True)

            list_segments = []
            last_pos = 0
            accumated_inc = 0
            all_text = ""

            # Iterate over the response content as it arrives
            for line in r.iter_lines():
                # Decode JSON string to dictionary
                data = json.loads(line)

                # process depending on returned information
                if "TranscriptionInfo" in data: # process info
                    info = convertToNamedTuple('TranscriptionInfo', data["TranscriptionInfo"])

                    language_name = LANGUAGES[info.language].title()
                    if not live:
                        print(
                            "Detected language '%s' with probability %f"
                            % (language_name, info.language_probability)
                        )

                    # Initialize tqdm progress bar when info is received
                    pbar = tqdm.tqdm(
                        total=info.duration, unit="seconds", disable=verbose or live is not False
                    )

                elif "Segment" in data: # process segment
                    segment = convertToNamedTuple('Segment', data["Segment"])

                    start, end, text = segment.start, segment.end, segment.text
                    all_text += segment.text

                    if verbose or options.print_colors:
                        if options.print_colors and segment.words:
                            text = self._get_colored_text(segment.words)
                        else:
                            text = segment.text

                        if not live:
                            line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
                            print(make_safe(line))

                    segment_dict = segment._asdict()
                    if segment.words:
                        segment_dict["words"] = [word._asdict() for word in segment.words]

                    list_segments.append(segment_dict)
                    duration = segment.end - last_pos
                    increment = (
                        duration
                        if accumated_inc + duration < info.duration
                        else info.duration - accumated_inc
                    )
                    accumated_inc += increment
                    last_pos = segment.end
                    # Update tqdm progress bar
                    if pbar:
                        pbar.update(increment)
            return dict(
                text=all_text,
                segments=list_segments,
                language=language_name,
            )
        except requests.exceptions.RequestException as e:
            # Handle any request exceptions, such as a connection error or timeout
            print(f"Error connecting to the remote URL:{self.remote_url}", e)
            return None