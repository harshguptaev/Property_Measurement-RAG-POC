import base64
import json
import io
import os
from typing import Dict, List, Optional, Tuple

from PIL import Image
import boto3
from botocore.exceptions import ClientError
import numpy as np
import faiss


EXPECTED_ORDER = ["top", "east", "west", "north", "south"]
EXPECTED_SIZE = (400, 400)


def _load_and_normalize_image(path: str, expected_size: Tuple[int, int] = EXPECTED_SIZE) -> Optional[Image.Image]:
    if not os.path.isfile(path):
        return None
    img = Image.open(path)
    if img.size != expected_size:
        img = img.resize(expected_size, Image.LANCZOS)
    return img.convert("RGB")


def stitch_pictometry_images_for_folder(folder_path: str, output_name: str = "stitched_image.png") -> Optional[str]:
    """
    Stitch five images (top, east, west, north, south) horizontally in that order
    and save as output_name within the same folder. Returns the saved path, or None
    if any of the required images are missing.
    """
    images: List[Image.Image] = []
    missing: List[str] = []
    for name in EXPECTED_ORDER:
        # try common extensions
        candidates = [
            os.path.join(folder_path, f"{name}.jpg"),
            os.path.join(folder_path, f"{name}.jpeg"),
            os.path.join(folder_path, f"{name}.png"),
            os.path.join(folder_path, f"{name}.webp"),
        ]
        loaded: Optional[Image.Image] = None
        for p in candidates:
            loaded = _load_and_normalize_image(p)
            if loaded is not None:
                break
        if loaded is None:
            missing.append(name)
        images.append(loaded)

    if any(img is None for img in images):
        return None

    total_width = EXPECTED_SIZE[0] * len(images)
    height = EXPECTED_SIZE[1]
    stitched = Image.new("RGB", (total_width, height), color=(0, 0, 0))

    x_offset = 0
    for img in images:
        stitched.paste(img, (x_offset, 0))
        x_offset += EXPECTED_SIZE[0]

    output_path = os.path.join(folder_path, output_name)
    stitched.save(output_path, format="PNG")
    
    top_image = images[0]
    # Save a compressed variant for storage/embedding (WEBP) and write base64 from it
    compressed_path = os.path.join(folder_path, "stitched_image.webp")
    top_image_compressed_path = os.path.join(folder_path, "top_image.webp")
    try:
        stitched.save(compressed_path, format="WEBP", quality=80, method=6)
        top_image.save(top_image_compressed_path, format="WEBP", quality=80, method=6)
    except Exception:
        compressed_path = output_path

    # Also save base64 string alongside the stitched image (use compressed if available)
    try:
        b64 = image_file_to_base64(compressed_path)
        b64_top_image = image_file_to_base64(top_image_compressed_path)
        b64_path = os.path.join(folder_path, "stiched_image_base64")
        b64_top_image_path = os.path.join(folder_path, "top_image_base64")
        with open(b64_path, "w", encoding="utf-8") as f:
            f.write(b64)
        with open(b64_top_image_path, "w", encoding="utf-8") as f:
            f.write(b64_top_image)
        # Generate and persist embeddings
        try:
            generate_image_embedings(folder_path, base64_string=b64)
            generate_image_embedings(folder_path, base64_string=b64_top_image)
        except Exception:
            pass
    except Exception:
        # If base64 persistence fails, we still return the image path
        pass

    return output_path


def stitch_all_pictometry_directories(root: str = None) -> Dict[str, Optional[str]]:
    """
    Iterate through pictometry_images/* and stitch images for each lat_lon folder.
    Returns a mapping of folder path to stitched image path (or None if skipped).
    """
    if root is None:
        root = os.path.join(os.getcwd(), "pictometry_images")
    if not os.path.isdir(root):
        return {}

    results: Dict[str, Optional[str]] = {}
    for name in os.listdir(root):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        result = stitch_pictometry_images_for_folder(folder)
        results[folder] = result
    return results


def image_file_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def generate_image_embedings(
    folder_path: str,
    *,
    base64_string: Optional[str] = None,
    region_name: str = "us-east-1",
    model_id: str = "amazon.titan-embed-image-v1",
    output_embedding_length: int = 1024,
) -> Optional[str]:
    """
    Generate image embeddings using Amazon Bedrock Titan Multimodal Embeddings G1
    for the stitched image in the given folder. Saves embeddings to
    "stiched_image_embedings" and returns the file path.

    If base64_string is not provided, it is read from "stiched_image_base64" in
    the same folder.
    """
    b64_path = os.path.join(folder_path, "stiched_image_base64")
    b64_top_image_path = os.path.join(folder_path, "top_image_base64")
    if base64_string is None:
        if not os.path.isfile(b64_path):
            return None
        with open(b64_path, "r", encoding="utf-8") as f:
            base64_string = f.read().strip()
    if not os.path.isfile(b64_top_image_path):
        return None
    with open(b64_top_image_path, "r", encoding="utf-8") as f:
        base64_string_top_image = f.read().strip()
    client = boto3.client("bedrock-runtime", region_name=region_name)
    payload = {
        "inputImage": base64_string,
        "embeddingConfig": {"outputEmbeddingLength": int(output_embedding_length)},
    }
    payload_top_image = {
        "inputImage": base64_string_top_image,
        "embeddingConfig": {"outputEmbeddingLength": int(output_embedding_length)},
    }
    response = client.invoke_model(modelId=model_id, body=json.dumps(payload))
    body = json.loads(response["body"].read())
    response_top_image = client.invoke_model(modelId=model_id, body=json.dumps(payload_top_image))
    body_top_image = json.loads(response_top_image["body"].read())
    embedding = (
        body.get("embedding")
        or (body.get("results") or [{}])[0].get("embedding")
        or (body.get("embeddings") or [None])[0]
    )
    embedding_top_image = (
        body_top_image.get("embedding")
        or (body_top_image.get("results") or [{}])[0].get("embedding")
        or (body_top_image.get("embeddings") or [None])[0]
    )
    if embedding is None or embedding_top_image is None:
        raise ValueError(f"Unable to extract embedding from response: {body}")

    out_path = os.path.join(folder_path, "stiched_image_embedings")
    out_path_top_image = os.path.join(folder_path, "top_image_embedings")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(embedding, f)
    with open(out_path_top_image, "w", encoding="utf-8") as f:
        json.dump(embedding_top_image, f)
    return out_path, out_path_top_image


def push_image_embedings_todb(
    folder_path: str,
    *,
    index_name: str = "pictometry_images",
    persist_directory: str = "vectorstore_faiss",
) -> Optional[str]:
    """
    Minimal direct FAISS integration using faiss-cpu.
    - Reads vector from stiched_image_embedings (JSON array of floats)
    - Creates/updates L2 index at vectorstore_faiss/{index_name}.faiss
    - Appends metadata to vectorstore_faiss/{index_name}_meta.jsonl
    Returns index path or None on failure.
    """
    embed_path = os.path.join(folder_path, "stiched_image_embedings")
    if not os.path.isfile(embed_path):
        return None
    try:
        with open(embed_path, "r", encoding="utf-8") as f:
            embedding = json.load(f)
        vec = np.asarray(embedding, dtype="float32").reshape(1, -1)
        if vec.size == 0:
            return None
    except Exception:
        return None

    # Prepare metadata (kept lightweight; we don't store base64)
    folder_name = os.path.basename(folder_path)
    try:
        lat_str, lon_str = folder_name.split("_", 1)
    except ValueError:
        lat_str, lon_str = "?", "?"
    text = (
        f"Stiched image in order (top,east,west,north,south) fetched from pictometry "
        f"service for lat:{lat_str} lon:{lon_str}"
    )
    metadata = {
        "lat": lat_str,
        "lon": lon_str,
        "folder": folder_path,
        "text": text,
    }

    os.makedirs(persist_directory, exist_ok=True)
    index_path = os.path.join(persist_directory, f"{index_name}.faiss")
    meta_path = os.path.join(persist_directory, f"{index_name}_meta.jsonl")

    # Load or create index
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        if index.d != vec.shape[1]:
            # Dimension mismatch: create a new index
            index = faiss.IndexFlatL2(vec.shape[1])
    else:
        index = faiss.IndexFlatL2(vec.shape[1])

    before = index.ntotal
    index.add(vec)
    faiss.write_index(index, index_path)

    # Append metadata with the assigned id (by position)
    record = {"id": before, **metadata}
    try:
        with open(meta_path, "a", encoding="utf-8") as mf:
            mf.write(json.dumps(record) + "\n")
    except Exception:
        pass

    return index_path


def _embed_image_with_bedrock(
    image_path: str,
    *,
    region_name: str = "us-east-1",
    model_id: str = "amazon.titan-embed-image-v1",
    output_embedding_length: int = 1024,
) -> Optional[List[float]]:
    """
    Try embedding using raw file bytes (WEBP base64 preserved), to match how
    stored embeddings were created. If that fails, fall back to JPEG transcode.
    """
    client = boto3.client("bedrock-runtime", region_name=region_name)
    
    def _call_with_b64(b64str: str) -> Optional[List[float]]:
        try:
            payload = {
                "inputImage": b64str,
                "embeddingConfig": {"outputEmbeddingLength": int(output_embedding_length)},
            }
            print("[Bedrock] invoke_model: sending request...")
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json",
            )
            print("[Bedrock] invoke_model: response received")
            body = json.loads(response["body"].read())
            embedding = (
                body.get("embedding")
                or (body.get("results") or [{}])[0].get("embedding")
                or (body.get("embeddings") or [None])[0]
            )
            return embedding if isinstance(embedding, list) else None
        except ClientError as e:
            try:
                err = e.response.get("Error", {})
                print("[Bedrock] ClientError:", err.get("Code"), err.get("Message"))
            except Exception:
                print("[Bedrock] ClientError:", repr(e))
            return None
        except Exception as e:
            import traceback
            print("[Bedrock] invoke_model failed:", repr(e))
            traceback.print_exc()
            return None

    # 1) Try with raw file bytes (e.g., WEBP)
    b64_raw = image_file_to_base64(image_path)
    # print(b64_raw)
    emb = _call_with_b64(b64_raw)
    if emb is not None:
        return emb

    # 2) Fallback: transcode to JPEG in-memory
    # try:
    #     img = Image.open(image_path).convert("RGB")
    #     buffer = io.BytesIO()
    #     img.save(buffer, format="JPEG", quality=85, optimize=True)
    #     b64_jpg = base64.b64encode(buffer.getvalue()).decode("utf-8")
    #     return _call_with_b64(b64_jpg)
    # except Exception:
    #     return None


def find_similar_images(
    stitched_image_path: str,
    *,
    index_name: str = "pictometry_images",
    persist_directory: str = "vectorstore_faiss",
    top_k: int = 2,
    use_cosine: bool = True,
) -> List[Dict[str, object]]:
    """
    Given a path to a stitched image (webp), compute its embedding and return top-k
    similar items from our pictometry FAISS store. Prints the matches and returns them.
    """
    index_path = os.path.join(persist_directory, f"{index_name}.faiss")
    meta_path = os.path.join(persist_directory, f"{index_name}_meta.jsonl")
    if not os.path.isfile(index_path) or not os.path.isfile(meta_path):
        print("Vector store not found. Run push_image_embedings_todb first.")
        return []

    query_vec_list = _embed_image_with_bedrock(stitched_image_path)
    if not query_vec_list:
        print("Failed to compute embedding for query image.")
        return []
    query = np.asarray(query_vec_list, dtype="float32").reshape(1, -1)

    index = faiss.read_index(index_path)
    d = index.d
    if query.shape[1] != d:
        print(f"Dim mismatch: index d={d}, query d={query.shape[1]}")
        return []

    # Load metadata rows
    meta: List[Dict[str, object]] = []
    with open(meta_path, "r", encoding="utf-8") as mf:
        for line in mf:
            try:
                meta.append(json.loads(line))
            except Exception:
                continue

    if use_cosine:
        # Build a temporary IP index from reconstructed vectors after L2-normalization
        ntotal = index.ntotal
        xb = np.zeros((ntotal, d), dtype="float32")
        for i in range(ntotal):
            try:
                xb[i] = np.asarray(index.reconstruct(i), dtype="float32")
            except Exception:
                # If reconstruct not supported, fallback to L2 search
                use_cosine = False
                break
        if use_cosine:
            # Normalize both base and query
            def _normalize(x: np.ndarray) -> np.ndarray:
                norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
                return x / norms

            xb_n = _normalize(xb)
            q_n = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
            ip_index = faiss.IndexFlatIP(d)
            ip_index.add(xb_n)
            scores, ids = ip_index.search(q_n, top_k)
            # Convert inner product to cosine similarity directly
            sims = scores[0].tolist()
            idxs = ids[0].tolist()
            results = []
            for sim, idx in zip(sims, idxs):
                if idx < 0 or idx >= len(meta):
                    continue
                m = meta[idx]
                m_out = {**m, "cosine": float(sim)}
                results.append(m_out)
            for r in results:
                print(r)
            return results

    # Fallback: L2 search on the original index
    distances, ids = index.search(query, top_k)
    dists = distances[0].tolist()
    idxs = ids[0].tolist()
    results = []
    for dist, idx in zip(dists, idxs):
        if idx < 0 or idx >= len(meta):
            continue
        m = meta[idx]
        m_out = {**m, "l2": float(dist)}
        results.append(m_out)
    for r in results:
        print(r)
    return results

__all__ = [
    "stitch_pictometry_images_for_folder",
    "stitch_all_pictometry_directories",
    "image_file_to_base64",
    "generate_image_embedings",
    "push_image_embedings_todb",
    "find_similar_images",
]


