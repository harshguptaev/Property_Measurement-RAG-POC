"""
Image similarity with Milvus Lite
- Stores existing Titan image embeddings from pictometry_images/* top_image_embedings
- Computes a query image embedding (via Bedrock Titan, same logic as src/poc_image_similarity.py)
- Searches Milvus Lite for the top-k most similar using COSINE

Notes:
- Per request: ignore stiched_image_embedings; only use top_image_embedings for indexing.
- Collection name: pictometry_images_index (1024-dim COSINE)
"""

import os
import json
import logging
import argparse
from typing import Any, Dict, List, Optional

from pymilvus import MilvusClient
from tqdm import tqdm

# Reuse exact embedding logic from the POC to avoid divergence
try:
    from src.image_similarity_poc.poc_image_similarity import _embed_image_with_bedrock  # type: ignore
except Exception:  # Fallback if running as module without package context
    from poc_image_similarity import _embed_image_with_bedrock  # type: ignore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ImageMilvus:
    """
    Minimal image indexer/searcher over Milvus Lite for Pictometry images.
    - Index: 1024-dim vectors (Titan image embeddings) with COSINE metric
    - Metadata: lat, lon, folder, text
    """

    def __init__(
        self,
        *,
        milvus_uri: str = "./milvus_demo.db",
        collection_name: str = "pictometry_images_index",
        embedding_dim: int = 1024,
    ) -> None:
        self.milvus_client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.metric_type = "COSINE"

    def ensure_collection(self, *, drop_existing: bool = False) -> None:
        """Create the image collection if missing; optionally drop-and-recreate."""
        exists = self.milvus_client.has_collection(self.collection_name)
        if exists and drop_existing:
            logger.info("Dropping existing collection: %s", self.collection_name)
            self.milvus_client.drop_collection(self.collection_name)
            exists = False
        if not exists:
            logger.info(
                "Creating collection %s (dim=%d, metric=%s)",
                self.collection_name,
                self.embedding_dim,
                self.metric_type,
            )
        self.milvus_client.create_collection(
                collection_name=self.collection_name,
                dimension=self.embedding_dim,
                metric_type=self.metric_type,
                consistency_level="Bounded",
            )
        logger.info("✅ Created Image collection: %s", self.collection_name)

    def index_top_embeddings(self, root_dir: str = "pictometry_images") -> int:
        """
        Traverse root_dir expecting subfolders named like "<lat>_<lon>". For each, read
        top_image_embedings (JSON array of floats, 1024-dim) and insert batch records.
        Returns number of inserted items.
        """
        if not os.path.isdir(root_dir):
            logger.error("Image root directory not found: %s", root_dir)
            return 0

        self.ensure_collection(drop_existing=True)

        records: List[Dict[str, Any]] = []
        counter = 0

        subfolders = [
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        ]

        for sub in tqdm(subfolders, desc="Indexing top_image_embedings"):
            try:
                embed_path = os.path.join(sub, "top_image_embedings")
                if not os.path.isfile(embed_path):
                    continue
                with open(embed_path, "r", encoding="utf-8") as f:
                    embedding = json.load(f)
                if not isinstance(embedding, list) or not embedding:
                    continue

                base = os.path.basename(sub)
                try:
                    lat_str, lon_str = base.split("_", 1)
                except ValueError:
                    lat_str, lon_str = "?", "?"

                text = (
                    "Top pictometry image embedding for lat:{lat} lon:{lon}".format(
                        lat=lat_str, lon=lon_str
                    )
                )

                records.append(
                    {
                    "id": counter,
                    "vector": embedding,
                    "lat": lat_str,
                    "lon": lon_str,
                    "folder": sub,
                    "text": text,
                    }
                )
                counter += 1
            except Exception as e:
                logger.error("Error processing %s: %s", sub, e)
                continue

        if not records:
            logger.warning("No top_image_embedings found under %s", root_dir)
            return 0

        try:
            self.milvus_client.insert(collection_name=self.collection_name, data=records)
            logger.info("✅ Inserted %d image vectors into %s", len(records), self.collection_name)
            return len(records)
        except Exception as e:
            logger.error("Failed inserting image data into Milvus: %s", e)
            return 0

    def embed_image(
        self,
        image_path: str,
        *,
        region_name: str = "us-east-1",
        model_id: str = "amazon.titan-embed-image-v1",
        output_embedding_length: int = 1024,
    ) -> Optional[List[float]]:
        """Compute Titan image embedding for a given image path (WEBP/JPG/PNG)."""
        if not os.path.isfile(image_path):
            logger.error("Image not found: %s", image_path)
            return None
        try:
            emb = _embed_image_with_bedrock(
                image_path,
                region_name=region_name,
                model_id=model_id,
                output_embedding_length=output_embedding_length,
            )
            if not emb:
                logger.error("Failed to compute embedding for %s", image_path)
                return None
            if len(emb) != self.embedding_dim:
                logger.warning(
                    "Embedding dim mismatch: got %d, expected %d",
                    len(emb),
                    self.embedding_dim,
                )
            return emb
        except Exception as e:
            logger.error("Embedding failed for %s: %s", image_path, e)
            return None

    def search_similar_by_image(
        self,
        image_path: str,
        *,
        top_k: int = 3,
        region_name: str = "us-east-1",
        model_id: str = "amazon.titan-embed-image-v1",
    ) -> List[Dict[str, Any]]:
        """
        Embed the query image and search Milvus for top-k most similar (COSINE).
        Returns a list of matches with metadata and similarity score.
        """
        self.ensure_collection(drop_existing=False)

        query_vec = self.embed_image(
            image_path,
            region_name=region_name,
            model_id=model_id,
            output_embedding_length=self.embedding_dim,
        )
        if not query_vec:
            return []

        try:
            results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[query_vec],
                limit=int(top_k),
                output_fields=["lat", "lon", "folder", "text", "reportId"],
            )
        except Exception as e:
            logger.error("Milvus search failed: %s", e)
            return []

        matches: List[Dict[str, Any]] = []
        if results and results[0]:
            for hit in results[0]:
                matches.append(
                    {
                        "id": hit.get("id"),
                        "lat": hit.get("lat"),
                        "lon": hit.get("lon"),
                        "reportId": hit.get("reportId"),
                        "folder": hit.get("folder"),
                        "text": hit.get("text"),
                        # For COSINE metric, Milvus returns similarity in [0,1]
                        "score": hit.get("distance"),
                    }
                )
        return matches


    def index_backgroundImage_embeddings(self, root_dir: str = "test_data") -> int:

        """
        Traverse root_dir expecting subfolders named like "reportid". For each, read
        backgroundImage_embedings (JSON array of floats, 1024-dim) and insert batch records.
        Returns number of inserted items.
        """
        if not os.path.isdir(root_dir):
            logger.error("Image root directory not found: %s", root_dir)
            return 0

        self.ensure_collection(drop_existing=True)

        records: List[Dict[str, Any]] = []
        counter = 0

        subfolders = [
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        ]

        for sub in tqdm(subfolders, desc="Indexing backgroundImage_embedings"):
            try:
                embed_path = os.path.join(sub, "backgroundImage_embedings")
                if not os.path.isfile(embed_path):
                    continue
                with open(embed_path, "r", encoding="utf-8") as f:
                    embedding = json.load(f)
                if not isinstance(embedding, list) or not embedding:
                    continue

                base = os.path.basename(sub)

                text = (
                    f"Ortho Image embedding for reportId: {base}"
                )

                records.append(
                    {
                    "id": counter,
                    "vector": embedding,
                    "reportId": base,
                    "folder": sub,
                    "text": text,
                    }
                )
                counter += 1
            except Exception as e:
                logger.error("Error processing %s: %s", sub, e)
                continue

        if not records:
            logger.warning("No backgroundImage_embedings found under %s", root_dir)
            return 0

        try:
            self.milvus_client.insert(collection_name=self.collection_name, data=records)
            logger.info("✅ Inserted %d image vectors into %s", len(records), self.collection_name)
            return len(records)
        except Exception as e:
            logger.error("Failed inserting image data into Milvus: %s", e)
            return 0



def print_matches(matches: List[Dict[str, Any]]) -> None:
    if not matches:
        print("No matches found.")
        return
    for i, m in enumerate(matches, 1):
        print(f"{i}. lat={m.get('lat')} lon={m.get('lon')} score={m.get('score'):.4f}")
        print(f"   folder: {m.get('folder')}")
        print(f"   text:   {m.get('text')}")