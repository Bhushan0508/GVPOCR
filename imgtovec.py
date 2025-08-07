import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType, utility

# -----------------------------------------------------------
# Part 1: Image to Vector Conversion (ResNet Embedding)
# -----------------------------------------------------------

# Load a pre-trained ResNet50 model
# 'pretrained=True' downloads the weights trained on ImageNet
model = models.resnet50(pretrained=True)

# Important: Modify the model to remove the final classification layer.
# We want the features (embeddings), not the classification probabilities.
# The 'fc' (fully connected) layer is the last one in ResNet responsible for classification.
# By taking all children up to the last one, we get the feature extractor.
model = torch.nn.Sequential(*(list(model.children())[:-1]))

# Set the model to evaluation mode. This disables dropout and batch normalization
# updates, which is crucial for consistent output during inference.
model.eval()

print("ResNet50 model loaded and configured for feature extraction.")
# The output dimension for ResNet50 before the fc layer is 2048.
VECTOR_DIMENSION = 2048 # This will be the dimension of vectors stored in Milvus

preprocess = transforms.Compose([
    transforms.Resize(256),         # Resize the image to 256x256 pixels
    transforms.CenterCrop(224),     # Crop the center to 224x224 (ResNet's expected input size)
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor (values between 0-1)
    # Normalize with ImageNet's mean and std dev. This is crucial because the model
    # was trained with data normalized this way.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def image_to_resnet_vector(image_path):
    """
    Converts an image file to a ResNet vector embedding.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: A 1D NumPy array representing the image vector.
    """
    try:
        # Load the image
        img = Image.open(image_path).convert("RGB") # Ensure it's in RGB format

        # Preprocess the image
        img_tensor = preprocess(img)
        # Add a batch dimension (e.g., [C, H, W] -> [1, C, H, W])
        img_tensor = img_tensor.unsqueeze(0)

        # Generate embedding
        with torch.no_grad(): # Disable gradient calculation to save memory and speed up inference
            embedding_tensor = model(img_tensor)

        # Convert the PyTorch tensor to a 1D NumPy array
        # .flatten() reshapes it to 1D, .numpy() converts to NumPy array
        image_vector = embedding_tensor.flatten().numpy()

        # It's common practice to L2 normalize vectors before storing them in Milvus
        # This helps in distance calculations (e.g., cosine similarity).
        image_vector = image_vector / np.linalg.norm(image_vector)

        return image_vector

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred during image processing: {e}")
        return None

# Example usage:
# Create a dummy image file for demonstration if you don't have one
try:
    from PIL import ImageDraw
    dummy_img = Image.new('RGB', (224, 224), color = 'red')
    d = ImageDraw.Draw(dummy_img)
    d.text((10,10), "Hello", fill=(255,255,0))
    dummy_img.save("test_image_1.jpg")

    dummy_img = Image.new('RGB', (224, 224), color = 'blue')
    d = ImageDraw.Draw(dummy_img)
    d.text((10,10), "World", fill=(255,0,255))
    dummy_img.save("test_image_2.jpg")

    print("Dummy images 'test_image_1.jpg' and 'test_image_2.jpg' created.")
except ImportError:
    print("Pillow not fully installed for ImageDraw. Please ensure you have test_image_1.jpg and test_image_2.jpg manually for the next steps.")


image_path_1 = "test_image_1.jpg" # Replace with your actual image path
image_vector_1 = image_to_resnet_vector(image_path_1)

if image_vector_1 is not None:
    print(f"Vector for '{image_path_1}': Shape {image_vector_1.shape}, Sample: {image_vector_1[:5]}...")

image_path_2 = "test_image_2.jpg" # Another image
image_vector_2 = image_to_resnet_vector(image_path_2)

if image_vector_2 is not None:
    print(f"Vector for '{image_path_2}': Shape {image_vector_2.shape}, Sample: {image_vector_2[:5]}...")

################################################################################################################

# For Milvus Lite (embedded):
milvus_client = MilvusClient() # By default stores data in ./milvus_data.db

# For Milvus Standalone (Docker) or Distributed:
try:
    milvus_client = MilvusClient(uri="tcp://localhost:19530",token="root:Milvus") # Replace with your Milvus URI
except Exception as ex:
    print(ex)

COLLECTION_NAME = "image_embeddings_resnet"

# Define the fields for your collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
]

# Define the collection schema
schema = CollectionSchema(fields, description="Image embeddings generated by ResNet")

# Drop collection if it already exists (for clean restarts in testing)
try:
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"Dropped existing collection: {COLLECTION_NAME}")
except Exception as ex:
    print(ex)

# Create the collection
try:
    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        # Consistency level: 'Bounded' for good balance of performance and consistency
        consistency_level="Bounded"
    )
except Exception as ex:
    print(ex)

print(f"Collection '{COLLECTION_NAME}' created with dimension {VECTOR_DIMENSION}.")

# Create a HNSW index for efficient similarity search
# HNSW is a good general-purpose index for high-dimensional vectors
# You can experiment with other index types like IVF_FLAT, IVF_SQ8
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE", # Or "L2" (Euclidean), "IP" (Inner Product) depending on your normalization
    params={"M": 8, "efConstruction": 64} # HNSW specific parameters
)
milvus_client.create_index(
    collection_name=COLLECTION_NAME,
    index_params=index_params
)
print(f"Index created on 'vector' field with type HNSW and COSINE metric.")


if image_vector_1 is not None and image_vector_2 is not None:
    # Prepare data for insertion
    # Each dictionary in the list represents one entity (row)
    data_to_insert = [
        {"image_path": image_path_1, "vector": image_vector_1.tolist()}, # .tolist() converts numpy array to Python list
        {"image_path": image_path_2, "vector": image_vector_2.tolist()}
    ]

    # Insert data
    res_insert = milvus_client.insert(
        collection_name=COLLECTION_NAME,
        data=data_to_insert
    )
    print(f"Inserted data into Milvus: {res_insert}")

    # Ensure data is loaded into memory for searching (for smaller collections, Milvus often loads automatically)
    milvus_client.load_collection(collection_name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' loaded into memory.")

    # -----------------------------------------------------------
    # Part 3: Perform a Similarity Search (Optional)
    # -----------------------------------------------------------

    # Let's search for images similar to image_1 itself
    query_vector = image_vector_1.tolist() # Or any new image vector you generate

    # Perform the search
    search_results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector], # Query data must be a list of vectors
        limit=2, # Get top 2 most similar results
        output_fields=["image_path"], # Return the image_path metadata field
        search_params={
            "metric_type": "COSINE",
            "params": {"ef": 64} # HNSW search parameter
        }
    )

    print("\nSearch Results:")
    for result_set in search_results: # search_results is a list of result sets (one per query vector)
        for hit in result_set:
            print(f"  ID: {hit['id']}, Distance (1-Similarity): {hit['distance']:.4f}, Image Path: {hit['entity']['image_path']}")

# Close the Milvus client (important for Milvus Lite to clean up resources, not strictly needed for Standalone/Distributed)
milvus_client.close()
print("Milvus client closed.")

