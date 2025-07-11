import torch
import requests
import numpy as np
from PIL import Image
from torchvision import transforms
from opensearchpy import OpenSearch
from io import BytesIO
import json
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])
client = OpenSearch(
    hosts=[{'host': 'superbotics-search-8731633774.eu-central-1.bonsaisearch.net', 'port': 443}],
    http_auth=('JC8YvqRN74', 'T8JEbusxGFN5VXZ'),
    use_ssl=True,
    verify_certs=True
)
index = "products_new"
query = {
    "size": 5000,
    "_source": ["image", "title", "price", "type"],
    "query": {"exists": {"field": "image"}}
}
response = client.search(index=index, body=query)
docs = response['hits']['hits']
vectors = []
metadata = []
def get_embedding(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        return model(img_tensor).squeeze().numpy().tolist()
for doc in docs:
    img_url = doc['_source']['image']
    try:
        res = requests.get(img_url)
        img = Image.open(BytesIO(res.content)).convert("RGB")
        vec = get_embedding(img)
        vectors.append(vec)
        metadata.append({
            "image": img_url,
            "title": doc['_source'].get('title'),
            "price": doc['_source'].get('price'),
            "type": doc['_source'].get('type')
        })
        print(f"✓ Indexed: {img_url}")
    except Exception as e:
        print(f"✗ Failed {img_url} - {e}")
np.save("embeddings.npy", np.array(vectors))
with open("metadata.json", "w") as f:
    json.dump(metadata, f)