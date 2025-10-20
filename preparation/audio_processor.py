# pylint: disable=C0301
'''
This module contains the AudioProcessor class and related functions for processing audio data.
It utilizes various libraries and models to perform tasks such as preprocessing, feature extraction,
and audio separation. The class is initialized with configuration parameters and can process
audio files using the provided models.
'''
import math
import os
import subprocess

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from audio_separator.separator import Separator
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput


class Wav2VecModel(Wav2Vec2Model):
    """
    Wav2VecModel is a custom model class that extends the Wav2Vec2Model class from the transformers library. 
    It inherits all the functionality of the Wav2Vec2Model and adds additional methods for feature extraction and encoding.
    ...

    Attributes:
        base_model (Wav2Vec2Model): The base Wav2Vec2Model object.

    Methods:
        forward(input_values, seq_len, attention_mask=None, mask_time_indices=None
        , output_attentions=None, output_hidden_states=None, return_dict=None):
            Forward pass of the Wav2VecModel. 
            It takes input_values, seq_len, and other optional parameters as input and returns the output of the base model.

        feature_extract(input_values, seq_len):
            Extracts features from the input_values using the base model.

        encode(extract_features, attention_mask=None, mask_time_indices=None, output_attentions=None, output_hidden_states=None, return_dict=None):
            Encodes the extracted features using the base model and returns the encoded features.
    """
    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass of the Wav2Vec model.

        Args:
            self: The instance of the model.
            input_values: The input values (waveform) to the model.
            seq_len: The sequence length of the input values.
            attention_mask: Attention mask to be used for the model.
            mask_time_indices: Mask indices to be used for the model.
            output_attentions: If set to True, returns attentions.
            output_hidden_states: If set to True, returns hidden states.
            return_dict: If set to True, returns a BaseModelOutput instead of a tuple.

        Returns:
            The output of the Wav2Vec model.
        """
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        """
        Extracts features from the input values and returns the extracted features.

        Parameters:
        input_values (torch.Tensor): The input values to be processed.
        seq_len (torch.Tensor): The sequence lengths of the input values.

        Returns:
        extracted_features (torch.Tensor): The extracted features from the input values.
        """
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        return extract_features

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Encodes the input features into the output space.

        Args:
            extract_features (torch.Tensor): The extracted features from the audio signal.
            attention_mask (torch.Tensor, optional): Attention mask to be used for padding.
            mask_time_indices (torch.Tensor, optional): Masked indices for the time dimension.
            output_attentions (bool, optional): If set to True, returns the attention weights.
            output_hidden_states (bool, optional): If set to True, returns all hidden states.
            return_dict (bool, optional): If set to True, returns a BaseModelOutput instead of the tuple.

        Returns:
            The encoded output features.
        """
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def linear_interpolation(features, seq_len):
    """
    Transpose the features to interpolate linearly.

    Args:
        features (torch.Tensor): The extracted features to be interpolated.
        seq_len (torch.Tensor): The sequence lengths of the features.

    Returns:
        torch.Tensor: The interpolated features.
    """
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


def resample_audio(input_audio_file: str, output_audio_file: str, sample_rate: int):
    p = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error", "-i", input_audio_file, "-ar", str(sample_rate), output_audio_file
    ])
    ret = p.wait()
    assert ret == 0, "Resample audio failed!"
    return output_audio_file


class AudioProcessor:
    """
    AudioProcessor is a class that handles the processing of audio files.
    It takes care of preprocessing the audio files, extracting features
    using wav2vec models, and separating audio signals if needed.

    :param sample_rate: Sampling rate of the audio file
    :param fps: Frames per second for the extracted features
    :param wav2vec_model_path: Path to the wav2vec model
    :param audio_separator_model_path: Path to the audio separator model
    :param audio_separator_model_name: Name of the audio separator model
    :param cache_dir: Directory to cache the intermediate results
    :param device: Device to run the processing on
    :param only_last_features: Whether to only use the last features
    """
    def __init__(
        self,
        sample_rate,
        fps,
        wav2vec_model_path,
        audio_separator_model_path:str=None,
        audio_separator_model_name:str=None,
        cache_dir:str='',
        only_last_features:bool=False,
        device="cuda:0",
    ) -> None:
        self.sample_rate = sample_rate
        self.fps = fps
        self.device = device

        self.audio_encoder = Wav2VecModel.from_pretrained(wav2vec_model_path, local_files_only=True).to(device=device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.only_last_features = only_last_features

        if audio_separator_model_name is not None:
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError as _:
                print("Fail to create the output cache dir.")
            self.audio_separator = Separator(
                output_dir=cache_dir,
                output_single_stem="vocals",
                model_file_dir=audio_separator_model_path,
            )
            self.audio_separator.load_model(audio_separator_model_name)
            assert self.audio_separator.model_instance is not None, "Fail to load audio separate model."
        else:
            self.audio_separator=None
            print("Use audio directly without vocals seperator.")


        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_path, local_files_only=True)


    def preprocess(self, wav_file: str, clip_length: int=-1):
        """
        Preprocess a WAV audio file by separating the vocals from the background and resampling it to a 16 kHz sample rate.
        The separated vocal track is then converted into wav2vec2 for further processing or analysis.

        Args:
            wav_file (str): The path to the WAV file to be processed. This file should be accessible and in WAV format.

        Raises:
            RuntimeError: Raises an exception if the WAV file cannot be processed. This could be due to issues
                        such as file not found, unsupported file format, or errors during the audio processing steps.

        Returns:
            torch.tensor: Returns an audio embedding as a torch.tensor
        """
        if self.audio_separator is not None:
            # 1. separate vocals
            # TODO: process in memory
            outputs = self.audio_separator.separate(wav_file)
            if len(outputs) <= 0:
                raise RuntimeError("Audio separate failed.")

            vocal_audio_file = outputs[0]
            vocal_audio_name, _ = os.path.splitext(vocal_audio_file)
            vocal_audio_file = os.path.join(self.audio_separator.output_dir, vocal_audio_file)
            vocal_audio_file = resample_audio(vocal_audio_file, os.path.join(self.audio_separator.output_dir, f"{vocal_audio_name}-16k.wav"), self.sample_rate)
        else:
            vocal_audio_file=wav_file

        # 2. extract wav2vec features
        speech_array, sampling_rate = librosa.load(vocal_audio_file, sr=self.sample_rate)
        audio_feature = np.squeeze(self.wav2vec_feature_extractor(speech_array, sampling_rate=sampling_rate).input_values)
        seq_len = math.ceil(len(audio_feature) / self.sample_rate * self.fps)
        audio_length = seq_len

        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)

        if clip_length>0 and seq_len % clip_length != 0:
            audio_feature = torch.nn.functional.pad(audio_feature, (0, (clip_length - seq_len % clip_length) * (self.sample_rate // self.fps)), 'constant', 0.0)
            seq_len += clip_length - seq_len % clip_length
        audio_feature = audio_feature.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=seq_len, output_hidden_states=True)
        assert len(embeddings) > 0, "Fail to extract audio embedding"
        if self.only_last_features:
            audio_emb = embeddings.last_hidden_state.squeeze()
        else:
            audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = rearrange(audio_emb, "b s d -> s b d")

        audio_emb = audio_emb.cpu().detach()

        return audio_emb, audio_length

    def get_embedding(self, wav_file: str):
        """preprocess wav audio file convert to embeddings

        Args:
            wav_file (str): The path to the WAV file to be processed. This file should be accessible and in WAV format.

        Returns:
            torch.tensor: Returns an audio embedding as a torch.tensor
        """
        speech_array, sampling_rate = librosa.load(
            wav_file, sr=self.sample_rate)
        assert sampling_rate == 16000, "The audio sample rate must be 16000"
        audio_feature = np.squeeze(self.wav2vec_feature_extractor(
            speech_array, sampling_rate=sampling_rate).input_values)
        seq_len = math.ceil(len(audio_feature) / self.sample_rate * self.fps)

        audio_feature = torch.from_numpy(
            audio_feature).float().to(device=self.device)
        audio_feature = audio_feature.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.audio_encoder(
                audio_feature, seq_len=seq_len, output_hidden_states=True)
        assert len(embeddings) > 0, "Fail to extract audio embedding"

        if self.only_last_features:
            audio_emb = embeddings.last_hidden_state.squeeze()
        else:
            audio_emb = torch.stack(
                embeddings.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = rearrange(audio_emb, "b s d -> s b d")

        audio_emb = audio_emb.cpu().detach()

        return audio_emb

    def close(self):
        """
        TODO: to be implemented
        """
        return self

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
