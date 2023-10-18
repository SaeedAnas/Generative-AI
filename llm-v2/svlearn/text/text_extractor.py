# -------------------------------------------------------------------------------------------------
#  Copyright (c) 2023.  SupportVectors AI Lab
#  This code is part of the training material, and therefore part of the intellectual property.
#  It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#  Use is limited to the duration and purpose of the training at SupportVectors.
#
#  Author: Asif Qamar
# -------------------------------------------------------------------------------------------------
import logging as log
import uuid
from typing import Any

# Tika is a library that allows you to extract text from a file in one of the many formats it supports
from tika import parser, detector, language

from svlearn.common import *


# -------------------------------------------------------------------------------------------------
class TextExtractionError(SVError):
    """
    Exception raised when a text could not be extracted from a file

    :param path: name of the file from which text could not be extracted
    :param message: explanation of the error
    :rtype: object

     """

    def __init__(self, path, message=None):
        super.__init__(super, path, message)
        self.path = path
        if not self.message:
            self.message = (f'Could not extract text from the file. '
                            f'Check if it a Tika-supported document format: {self.path}')


# -------------------------------------------------------------------------------------------------

class TextExtraction:
    def __init__(self, path: str):
        self.path_ = path
        self.text_ = self.to_text(self.path_)
        self.doctype_ = self.document_type(self.path_)
        self.language_ = language.from_buffer(self.text_)
        self.id_ = str(uuid.uuid4())

    @staticmethod
    def to_text(path: str) -> str:
        """
        Extracts plain-text from a file, in one of the Tika-supported formats
        :param path: path to the document file
        :return: text from document file
        """
        # Preconditions check for an existing, readable, non-empty file
        check_valid_file(path)

        log.info(f'Parsing file: {path}')
        try:
            text_content: str = (parser.from_file(path))['content']
            if text_content is None:
                raise TextExtractionError(path=path, 
                message=f'No content found in file: {path}')
            return text_content.strip()
        except Exception:
            raise TextExtractionError(path)

    
    @staticmethod
    def document_type(path: str) -> str:
        """
        Determines the MIME type of the file
        :param path: the filesystem path to the document.
        :return: the MIME-type, such as "application/pdf"
        """
        # Preconditions check for an existing, readable, non-empty file
        check_valid_file(path)
        return detector.from_file(path)

    def __repr__(self):
        limit: int = min(100, len(self.text_))
        return f' Document type: {self.doctype_}\n Language: {self.language_}\n Text: {self.text_[:limit]}...'
