"""LLM client for local and API-based inference"""
import torch
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from loguru import logger
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMClient:
    """Unified LLM client supporting local and remote models"""
    
    def __init__(self, model_name: str, api_url: Optional[str] = None, 
                 device: str = "cuda", load_in_8bit: bool = True):
        """
        Initialize LLM client
        
        Args:
            model_name: HuggingFace model identifier
            api_url: Optional API endpoint for remote inference
            device: Device to use (cuda/cpu)
            load_in_8bit: Whether to use 8-bit quantization
        """
        self.model_name = model_name
        self.api_url = api_url
        self.device = device
        
        if api_url:
            logger.info(f"Using remote LLM API: {api_url}")
            self.model = None
            self.tokenizer = None
        else:
            logger.info(f"Loading local model: {model_name}")
            self._load_local_model(load_in_8bit)
    
    def _load_local_model(self, load_in_8bit: bool):
        """Load model locally with optional quantization"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Configure quantization
            if load_in_8bit and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            self.model.eval()
            logger.info(f"Model loaded on {self.device}")
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_async(self, prompt: str, max_tokens: int = 256, 
                             temperature: float = 0.3, top_p: float = 0.9) -> str:
        """Generate text asynchronously"""
        if self.api_url:
            return await self._generate_api(prompt, max_tokens, temperature, top_p)
        else:
            return self._generate_local(prompt, max_tokens, temperature, top_p)
    
    def generate(self, prompt: str, max_tokens: int = 256, 
                 temperature: float = 0.3, top_p: float = 0.9) -> str:
        """Generate text synchronously"""
        if self.api_url:
            raise ValueError("Use generate_async for API-based inference")
        return self._generate_local(prompt, max_tokens, temperature, top_p)
    
    def _generate_local(self, prompt: str, max_tokens: int, 
                        temperature: float, top_p: float) -> str:
        """Generate using local model"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""
    
    async def _generate_api(self, prompt: str, max_tokens: int, 
                            temperature: float, top_p: float) -> str:
        """Generate using remote API"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_url}/completions",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["text"]
        
        except Exception as e:
            logger.error(f"API generation error: {e}")
            return ""
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate for multiple prompts"""
        return [self.generate(prompt, **kwargs) for prompt in prompts]