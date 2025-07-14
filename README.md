This is a Flask web application that allows users to upload a product image and find visually similar items using Meta AI's DINOv2 model and OpenSearch. The application features category-based search, pagination, and a modern Tailwind CSS frontend.

#Features
1.Upload product image to find similar items.
2.Embedding extraction using DINOv2 (ViT-G/14).
3.Visual similarity search using cosine similarity.
4.All products page with pagination (20 per page).
5.Powered by OpenSearch for scalable indexing and querying.
6.Clean UI built with Tailwind CSS.

#How It Works
1.Image Upload: User uploads an image via index.html.
2.Embedding Extraction: The image is processed using DINOv2 to get a feature vector.
3.Similarity Search: Cosine similarity is used to compare the query vector with embeddings.npy.
4.Display Results: The top 5 most similar products are shown on results.html.
5.Browse All: /all-products displays all products from OpenSearch, paginated (20 per page).

#Dependencies
Flask
Torch (PyTorch)
Torchvision
Pillow
NumPy
scikit-learn
requests
opensearch-py
Tailwind CSS (via CDN)

#Project Structure
.
├── app.py # Flask web app for handling uploads, search, and product listing
├── vector.py # Script to extract DINOv2 embeddings and metadata from OpenSearch
├── embeddings.npy # Numpy array of image embeddings
├── metadata.json # Metadata (image URL, title, type, price)
├── templates/
│ ├── index.html # Home upload page
│ ├── results.html # Results display after search
│ └── all_products.html# Paginated product grid view
├── static/
│ └── bg.jpg # background image for UI
├── .env # Environment variables for OpenSearch access

#Install Dependencies
pip install flask pillow numpy torch torchvision scikit-learn opensearch-py requests

