from typing import Dict, Any
from langchain_aws import BedrockLLM

class ChatModel:
    def __init__(self, model_id: str, model_kwargs: Dict[str, Any]):
        """
        Initialize the chat model with Bedrock
        
        Args:
            model_id (str): The model ID to use (e.g. 'us.anthropic.claude-3-7-sonnet-20250219-v1:0')
            model_kwargs (Dict[str, Any]): Model parameters like temperature, max_tokens etc.
        """
        self.model_id = model_id
        self.model_kwargs = model_kwargs
        self.llm = BedrockLLM(
            model_id=model_id,
            model_kwargs=model_kwargs
        ) 