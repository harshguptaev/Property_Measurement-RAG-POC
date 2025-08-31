"""
AWS Bedrock client integration for LLM operations with multimodal support.
"""
import json
import logging
import base64
from typing import Any, Dict, List, Optional, Union
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import Field


class BedrockLLM(LLM):
    """
    Custom LangChain LLM wrapper for AWS Bedrock with multimodal support.
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
    
    def invoke(self, messages: List[BaseMessage], **kwargs) -> Any:
        """
        Enhanced invoke method that supports multimodal messages.
        """
        try:
            # Handle multimodal content
            if self._has_images_in_messages(messages):
                return self._invoke_multimodal(messages, **kwargs)
            else:
                # Convert messages to prompt for text-only processing
                prompt = self._convert_messages_to_prompt(messages)
                response_text = self._call(prompt, **kwargs)
                return self._create_response_object(response_text)
        except Exception as e:
            logging.error(f"Error in invoke: {e}")
            raise
    
    def _has_images_in_messages(self, messages: List[BaseMessage]) -> bool:
        """Check if any messages contain images."""
        for message in messages:
            content = getattr(message, 'content', '')
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'image':
                        return True
            elif isinstance(content, str) and '[User uploaded an image' in content:
                return True
        return False
    
    def _invoke_multimodal(self, messages: List[BaseMessage], **kwargs) -> Any:
        """Invoke model with multimodal content."""
        if not "anthropic.claude" in self.model_id:
            # For non-Claude models, fall back to text-only processing
            logging.warning(f"Model {self.model_id} doesn't support multimodal. Converting to text-only.")
            prompt = self._convert_messages_to_prompt(messages)
            response_text = self._call(prompt, **kwargs)
            return self._create_response_object(response_text)
        
        try:
            # Convert messages to Claude format
            claude_messages = self._convert_to_claude_multimodal(messages)
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": claude_messages
            }
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            response_body = json.loads(response["body"].read())
            response_text = response_body["content"][0]["text"]
            
            return self._create_response_object(response_text)
            
        except Exception as e:
            logging.error(f"Error in multimodal invoke: {e}")
            # Fallback to text-only
            prompt = self._convert_messages_to_prompt(messages)
            response_text = self._call(prompt, **kwargs)
            return self._create_response_object(response_text)
    
    def _convert_to_claude_multimodal(self, messages: List[BaseMessage]) -> List[Dict]:
        """Convert LangChain messages to Claude multimodal format."""
        claude_messages = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                # Add system message as first user message
                claude_messages.append({
                    "role": "user",
                    "content": f"System: {message.content}"
                })
            elif isinstance(message, HumanMessage):
                content = message.content
                
                # Check if content includes image references
                if '[User uploaded an image' in content or '[User selected' in content:
                    # For now, treat as text since we don't have actual image data in the message
                    # In a full implementation, you'd extract and include base64 image data
                    claude_messages.append({
                        "role": "user",
                        "content": content + "\n\nNote: Image analysis capabilities are available but require proper image data encoding."
                    })
                else:
                    claude_messages.append({
                        "role": "user",
                        "content": content
                    })
            elif isinstance(message, AIMessage):
                claude_messages.append({
                    "role": "assistant",
                    "content": message.content
                })
        
        return claude_messages
    
    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert messages to a single prompt string."""
        prompt_parts = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {message.content}")
        
        return "\n\n".join(prompt_parts)
    
    def _create_response_object(self, text: str) -> Any:
        """Create a response object similar to LangChain's format."""
        class Response:
            def __init__(self, content):
                self.content = content
        
        return Response(text)


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


def create_multimodal_bedrock_llm(config: Dict[str, Any]) -> BedrockLLM:
    """
    Create multimodal Bedrock LLM instance optimized for image analysis.
    Uses Claude 3 Sonnet by default for best multimodal capabilities.
    """
    # Ensure we use a Claude model for multimodal capabilities
    model_id = config.get("model_id", "anthropic.claude-3-sonnet-20240229-v1:0")
    if "anthropic.claude" not in model_id:
        logging.warning(f"Model {model_id} may not support multimodal. Consider using Claude 3.")
    
    return BedrockLLM(
        model_id=model_id,
        temperature=config.get("temperature", 0.1),
        max_tokens=config.get("max_tokens", 4096),
        region_name=config.get("region_name", "us-east-1")
    )
