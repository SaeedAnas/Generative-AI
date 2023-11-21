from sentence_transformers import SentenceTransformer
import torch

class TextEmbedder:
    def __init__(self, model_name):
        if torch.cuda.is_available():
            device = "cuda"
        else: 
            device = "cpu"
        
        self.model = SentenceTransformer(model_name, device=device)
    
    def embed(self, sentences, batch_size=128):
        return self.model.encode(sentences, batch_size=batch_size, convert_to_numpy=True)
        