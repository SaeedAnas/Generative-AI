import logging as log
from typing import List

from numpy import ndarray
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import torch
from imagebind import data


from svlearn.common.svexception import SVError
from svlearn.config.configuration import ConfigurationMixin


class ImageBindEmbedder(ConfigurationMixin):

    def __init__(self):
        def __init__(self):
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model = imagebind_model.imagebind_huge(pretrained=True)
            self.model.eval()
            self.model.to(self.device)

    def batch_embed(self, paths: list[tuple[str, str]]):
        """
        Input: (modality type, path)
        modality type: text, vision, audio
        """
        inputs, mapping = self.format_input(paths)

        with torch.no_grad():
            embeddings = self.model.encode(inputs)

        outputs = self.format_output(embeddings, mapping)

        return outputs

    def format_input(self, paths: list[tuple[str, str]]):
        inputs = {}
        mapping = {}
        for idx, p in enumerate(paths):
            modality, path = p
            modality = self.parse_modality(modality)

            if modality not in inputs:
                inputs[modality] = []
                mapping[modality] = []

            inputs[modality].append(path)
            mapping[modality].append(idx)

        for modality in inputs:
            if modality == ModalityType.TEXT:
                inputs[modality] = data.load_and_transform_text(
                    inputs[modality], self.device)
            elif modality == ModalityType.VISION:
                inputs[modality] = data.load_and_transform_vision_data(
                    inputs[modality], self.device)
            elif modality == ModalityType.AUDIO:
                inputs[modality] = data.load_and_transform_audio_data(
                    inputs[modality], self.device)
            else:
                raise Exception("Invalid modality type")

        return inputs, mapping

    def parse_modality(self, modality: str):
        if modality == "text":
            return ModalityType.TEXT
        elif modality == "vision":
            return ModalityType.VISION
        elif modality == "audio":
            return ModalityType.AUDIO
        else:
            raise ValueError(f"Invalid modality type: {modality}")

    def format_output(self, embeddings, mapping):
        num_paths = sum([len(idxs) for idxs in mapping.values()])
        outputs = [None] * num_paths

        for (modality, idxs) in mapping.items():
            for idx, embedding in zip(idxs, embeddings[modality]):
                outputs[idx] = (modality, embedding)

        return outputs
