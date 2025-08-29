"""
AWS Bedrock client integration for LLM operations.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Union
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field


class BedrockLLM(LLM):
    """
    Custom LangChain LLM wrapper for AWS Bedrock.
    """
    
    client: Any = Field(default=None, exclude=True)
    model_id: str = Field(default="anthropic.claude-3-sonnet-20240229-v1:0")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=4096)
    region_name: str = Field(default="us-east-1")
    
    class Config:
        """Configuration for this pydantic object."""
        extra = "forbid"
    
    def __init__(self, **kwargs):
        """Initialize Bedrock LLM."""
        super().__init__(**kwargs)
        self._setup_client()
    
    def _setup_client(self):
        """Setup Bedrock runtime client."""
        try:
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.region_name
            )
            logging.info(f"Bedrock client initialized for region: {self.region_name}")
        except NoCredentialsError:
            logging.error("AWS credentials not found. Please configure your credentials.")
            raise
        except Exception as e:
            logging.error(f"Error initializing Bedrock client: {e}")
            raise
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "bedrock"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Bedrock model."""
        try:
            # Prepare the request body based on model type
            if "anthropic.claude" in self.model_id:
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            elif "amazon.titan" in self.model_id:
                body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": self.max_tokens,
                        "temperature": self.temperature,
                        "stopSequences": stop or []
                    }
                }
            else:
                # Generic format
                body = {
                    "prompt": prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
            
            # Make the request
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            # Parse response
            response_body = json.loads(response["body"].read())
            
            # Extract text based on model type
            if "anthropic.claude" in self.model_id:
                return response_body["content"][0]["text"]
            elif "amazon.titan" in self.model_id:
                return response_body["results"][0]["outputText"]
            else:
                # Try common response formats
                if "completion" in response_body:
                    return response_body["completion"]
                elif "text" in response_body:
                    return response_body["text"]
                else:
                    return str(response_body)
                    
        except ClientError as e:
            logging.error(f"Bedrock API error: {e}")
            raise
        except Exception as e:
            logging.error(f"Error calling Bedrock model: {e}")
            raise


class BedrockEmbeddings(Embeddings):
    """
    Custom LangChain Embeddings wrapper for AWS Bedrock.
    """
    
    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v1",
        region_name: str = "us-east-1",
        **kwargs
    ):
        """Initialize Bedrock embeddings."""
        super().__init__(**kwargs)
        self.model_id = model_id
        self.region_name = region_name
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup Bedrock runtime client."""
        try:
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.region_name
            )
            logging.info(f"Bedrock embeddings client initialized for region: {self.region_name}")
        except NoCredentialsError:
            logging.error("AWS credentials not found. Please configure your credentials.")
            raise
        except Exception as e:
            logging.error(f"Error initializing Bedrock embeddings client: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = []
        for text in texts:
            embedding = self._embed_text(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self._embed_text(text)
    
    def _embed_text(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        try:
            # Prepare request body based on model
            if "amazon.titan-embed" in self.model_id:
                body = {
                    "inputText": text
                }
            else:
                body = {
                    "texts": [text],
                    "input_type": "search_document"
                }
            
            # Make the request
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            # Parse response
            response_body = json.loads(response["body"].read())
            
            # Extract embedding based on model type
            if "amazon.titan-embed" in self.model_id:
                return response_body["embedding"]
            elif "cohere.embed" in self.model_id:
                return response_body["embeddings"][0]
            else:
                # Try common response formats
                if "embedding" in response_body:
                    return response_body["embedding"]
                elif "embeddings" in response_body:
                    return response_body["embeddings"][0]
                else:
                    logging.error(f"Unknown embedding response format: {response_body}")
                    raise ValueError("Unable to extract embedding from response")
                    
        except ClientError as e:
            logging.error(f"Bedrock embedding API error: {e}")
            raise
        except Exception as e:
            logging.error(f"Error getting embedding from Bedrock: {e}")
            raise


def create_bedrock_llm(config: Dict[str, Any]) -> BedrockLLM:
    """Create Bedrock LLM instance from configuration."""
    return BedrockLLM(
        model_id=config.get("model_id", "anthropic.claude-3-sonnet-20240229-v1:0"),
        temperature=config.get("temperature", 0.1),
        max_tokens=config.get("max_tokens", 4096),
        region_name=config.get("region_name", "us-east-1")
    )


def create_bedrock_embeddings(config: Dict[str, Any]) -> BedrockEmbeddings:
    """Create Bedrock embeddings instance from configuration."""
    return BedrockEmbeddings(
        model_id=config.get("embedding_model_id", "amazon.titan-embed-text-v1"),
        region_name=config.get("region_name", "us-east-1")
    )
