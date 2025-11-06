#!/usr/bin/env python3
"""
LLM Handler Module
Handles communication with multiple LLMs (Ollama, OpenAI, Gemini, Claude)
"""

import os
import yaml
import requests


class OllamaHandler:
    """Handler for Ollama (local LLM)"""
    
    def __init__(self, api_url, model, temperature=0.7):
        """Initialize Ollama handler"""
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
    
    def check_connection(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return True
            return False
        except:
            return False
    
    def generate(self, prompt):
        """Generate response from Ollama"""
        try:
            url = f"{self.api_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            return None
        
        except Exception as e:
            print(f"Error: {str(e)}")
            return None


class OpenAIHandler:
    """Handler for OpenAI"""
    
    def __init__(self, api_key, model="gpt-3.5-turbo", temperature=0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
    
    def check_connection(self):
        """Check if API key is valid"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            # Try a simple call to verify the key
            client.models.list()
            return True
        except Exception as e:
            print(f"Connection check error: {str(e)}")
            return False
    
    def generate(self, prompt):
        """Generate response from OpenAI"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI Error: {str(e)}")
            return None


class GeminiHandler:
    """Handler for Google Gemini"""
    
    def __init__(self, api_key, model="gemini-2.5-flash", temperature=0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
    
    def check_connection(self):
        """Check if API key is valid"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            return True
        except:
            return False
    
    def generate(self, prompt):
        """Generate response from Gemini"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt)
            
            return response.text
        except Exception as e:
            print(f"Gemini Error: {str(e)}")
            return None


class ClaudeHandler:
    """Handler for Anthropic Claude"""
    
    def __init__(self, api_key, model="claude-3-haiku-20240307", temperature=0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
    
    def check_connection(self):
        """Check if API key is valid"""
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key)
            return True
        except:
            return False
    
    def generate(self, prompt):
        """Generate response from Claude"""
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key)
            
            message = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text
        except Exception as e:
            print(f"Claude Error: {str(e)}")
            return None


class LLMHandler:
    """Main LLM Handler - routes to appropriate LLM"""
    
    def __init__(self, llm_type, api_key=None, config_path="config.yaml"):
        """Initialize LLM handler"""
        
        self.config = self.load_config(config_path)
        self.llm_type = llm_type
        self.api_key = api_key
        
        if llm_type == "ollama":
            if not self.config or 'llm' not in self.config:
                raise ValueError("Config not found or invalid!")
            
            ollama_config = self.config['llm'].get('ollama', {})
            
            if not ollama_config:
                raise ValueError("Ollama config not found in config.yaml!")
            
            self.llm = OllamaHandler(
                api_url=ollama_config.get('api_url', 'http://localhost:11434'),
                model=ollama_config.get('model', 'llama3.2:1b'),
                temperature=ollama_config.get('temperature', 0.7)
            )
        
        elif llm_type == "openai":
            self.llm = OpenAIHandler(api_key=api_key)
        
        elif llm_type == "gemini":
            self.llm = GeminiHandler(api_key=api_key)
        
        elif llm_type == "claude":
            self.llm = ClaudeHandler(api_key=api_key)
        
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    def load_config(self, config_path="config.yaml"):
        """Load configuration from YAML file"""
        
        if not os.path.exists(config_path):
            return None
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except:
            return None
    
    def check_connection(self):
        """Check if LLM is available"""
        return self.llm.check_connection()
    
    def generate_answer(self, prompt):
        """Generate answer from LLM"""
        return self.llm.generate(prompt)