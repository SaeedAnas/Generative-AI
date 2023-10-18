import os
import glob
import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

modality_info = {
    'image': {
        'type': ModalityType.VISION, 
        'folder': '/Volumes/MacX_1TB/devX/data/image/images/25kphotos_sample/', 
        #   'folder': '', 
        'extensions': ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    },
    'audio': {
        'type': ModalityType.AUDIO, 
        'folder': '/path/to/audio', 
        'extensions': ['*.wav', '*.mp3', '*.flac']
    }
}

def gather_media_files(folder, extensions):
    return [file for ext in extensions for file in glob.glob(os.path.join(folder, ext))]

def precompute_embeddings(modality_key):
    '''Compute embeddings for files of a specific modality.'''
    info = modality_info[modality_key]
    media_paths = gather_media_files(info['folder'], info['extensions'])
    
    if not media_paths:
        print(f'No files found for modality: {modality_key}\n')
        return {}

    modality = info['type']
    if modality == ModalityType.VISION:
        media_inputs = data.load_and_transform_vision_data(media_paths, device)
    elif modality == ModalityType.AUDIO:
        media_inputs = data.load_and_transform_audio_data(media_paths, device)

    with torch.no_grad():
        media_embeddings = model({modality: media_inputs})
    
    return dict(zip(media_paths, media_embeddings[modality].tolist()))

def retrieve_top_matches(query_embedding, media_embeddings, topk):
    '''Determine and return top matching media files based on similarity to a given embedding.'''
    similarities = {
        path: torch.nn.functional.cosine_similarity(query_embedding, torch.tensor(embedding, dtype=torch.float32).to(device), dim=0).item() 
        for path, embedding in media_embeddings.items()
    }

    return sorted(similarities, key=similarities.get, reverse=True)[:topk]

def text_query(query, topK, *modalities_to_search):
    '''Generate an embedding for a text query and retrieve top matches from specified modalities.'''
    text_input = data.load_and_transform_text([query], device)
    with torch.no_grad():
        query_embedding = model({ModalityType.TEXT: text_input})[ModalityType.TEXT][0]

    embeddings_map = {'image': image_embeddings, 'audio': audio_embeddings}
    result = {
        modality: retrieve_top_matches(query_embedding, embeddings_map[modality],topK) 
        for modality in modalities_to_search
    }
    return transform_output(result)

def transform_output(output):
    transformed_result = []
    
    for modality, paths in output.items():
        for path in paths:
            transformed_result.append({
                "type": modality,
                # "link": "http://localhost:8006/static/" + path.replace("/Volumes/MacX_1TB/devX/data/image/images/25kphotos_sample/","")
                "link": "http://localhost:8006/static/" + os.path.basename(path)
            })
    
    return {"result": transformed_result}

# Precompute and cache embeddings for faster retrieval
image_embeddings = precompute_embeddings('image')
audio_embeddings = precompute_embeddings('audio')


# results = text_query(query='hills', search='image', topk=10)
# print(results)
