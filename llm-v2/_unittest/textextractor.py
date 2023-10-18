import pytest
from svlearn.text import TextExtraction

# Sample test file paths
VALID_FILE_PATH = "./_unittest/testdata/valid_sample_sentences.txt"  # paper-llm-as-optimizer.pdf"#valid_sample_sentences.txt"
INVALID_FILE_PATH = "./_unittest/testdata/non_existent_file.txt"
EMPTY_FILE_PATH = "./_unittest/testdata/empty_file.txt"


# Create a fixture to initialize a TextExtraction instance for testing
@pytest.fixture
def text_extraction_instance():
    return TextExtraction(VALID_FILE_PATH)


# Test Initialization
def test_initialization_with_valid_file(text_extraction_instance):
    assert isinstance(text_extraction_instance, TextExtraction)


def test_initialization_with_invalid_file():
    with pytest.raises(FileNotFoundError):
        TextExtraction(INVALID_FILE_PATH)


# Test Text Extraction
def test_text_extraction(text_extraction_instance):
    instance = text_extraction_instance
    instance_text = instance.text_.split('\n')
    with open(VALID_FILE_PATH, 'r') as f:
        sample_text = f.read().split('\n')
    assert instance_text == sample_text


# def test_text_extraction_empty_file():
#     # with pytest.raises(TextExtractionError):
#     assert isinstance(text_extractor.TextExtraction(EMPTY_FILE_PATH),text_extractor.TextExtractionError)

# Test Document Type Detection
def test_document_type_detection(text_extraction_instance):
    instance = text_extraction_instance
    assert instance.doctype_ == "text/plain"  # "application/pdf"#"text/plain" #


def test_language_detection(text_extraction_instance):
    instance = text_extraction_instance
    assert instance.language_ == "en"


# Test UUID Generation
def test_uuid_generation():
    instance1 = TextExtraction(VALID_FILE_PATH)
    instance2 = TextExtraction(VALID_FILE_PATH)
    assert instance1.id_ != instance2.id_


# Test Exception Handling
def test_exception_handling_invalid_input():
    with pytest.raises(FileNotFoundError):
        TextExtraction(INVALID_FILE_PATH)


if __name__ == "__main__":
    pytest.main()
