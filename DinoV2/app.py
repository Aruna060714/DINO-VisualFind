from flask import Flask, request, render_template
from PIL import Image
import torch
import numpy as np
import json
from io import BytesIO
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from opensearchpy import OpenSearch
app = Flask(__name__)
client = OpenSearch(
    hosts=[{
        'host': 'superbotics-search-8731633774.eu-central-1.bonsaisearch.net',
        'port': 443
    }],
    http_auth=('JC8YvqRN74', 'T8JEbusxGFN5VXZ'),
    use_ssl=True,
    verify_certs=True
)
INDEX_NAME = "products_new"
PAGE_SIZE = 20
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])
vectors = np.load("embeddings.npy")
with open("metadata.json") as f:
    metadata = json.load(f)
def get_embedding(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        return model(img_tensor).squeeze().numpy().reshape(1, -1)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files['image']
    img = Image.open(uploaded_file.stream).convert('RGB')
    query_vector = get_embedding(img)
    similarities = cosine_similarity(query_vector, vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:5]
    top_results = []
    for idx in top_indices:
        item = metadata[idx]
        top_results.append({
            "image": item["image"],
            "title": item.get("title", "No Title"),
            "category": item.get("type", "Unknown"),
            "distance": round(float(similarities[idx]), 4)
        })

    return render_template("results.html", results=top_results, category="Auto", color="N/A")
@app.route('/all-products')
def all_products():
    page = int(request.args.get('page', 1))
    start_from = (page - 1) * PAGE_SIZE
    query = {
        "from": start_from,
        "size": PAGE_SIZE,
        "_source": ["image", "title", "price", "type"],
        "query": {
            "bool": {
                "must": [
                    {"exists": {"field": "image"}},
                    {
                        "script": {
                            "script": {
                                "source": "doc['image'].size() > 0 && doc['image'].value != ''",
                                "lang": "painless"
                            }
                        }
                    }
                ]
            }
        }
    }
    response = client.search(index=INDEX_NAME, body=query)
    docs = [hit["_source"] for hit in response["hits"]["hits"]]
    total_docs = response["hits"]["total"]["value"]
    total_pages = (total_docs + PAGE_SIZE - 1) // PAGE_SIZE
    return render_template(
        "all_products.html",
        products=docs,
        page=page,
        total_pages=total_pages,
        total=total_docs
    )
if __name__ == '__main__':
    app.run(debug=True)