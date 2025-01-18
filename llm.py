from openai import AsyncOpenAI
import re
import os
import asyncio
import numpy as np
import json
from pathlib import Path
from typing import List
from tqdm import tqdm

class LLMInterface:
    def __init__(self):
        # Get OpenAI API key from environment variable
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.model = 'gpt-4o'
        self.temperature = 0.7
        self.max_tokens = 500
        self.max_retries = 3
        
        print("Using standard OpenAI configuration")
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.openai.com/v1" 
        )
        
        print(f"LLM Interface initialized with:")
        print(f"- Model: {self.model}")
        print(f"- API Key present: {bool(self.api_key)}")
        print(f"- Base URL: {self.client.base_url}")


    async def generate_response(self, prompt: str, system_prompt: str = None, image_base64: str = None) -> str:
        """Base method for generating responses from the LLM"""
        if system_prompt is None:
            system_prompt = ""
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if image_base64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

    async def generate_embeddings(self, texts, batch_size=100):
        """Generate embeddings for a list of texts in batches"""
        try:
            all_embeddings = []
            
            # Clean all texts first
            cleaned_texts = []
            for text in texts:
                # Convert to string and basic cleaning
                text = str(text).strip()
                
                # Remove markdown-style formatting
                text = text.replace('**NAME:**', 'Name:')
                text = text.replace('**DESCRIPTION:**', 'Description:')
                text = text.replace('**', '')
                
                # Replace newlines with spaces and remove multiple spaces
                text = ' '.join(text.split())
                
                # Handle empty strings
                if not text:
                    text = "empty text"
                
                    
                cleaned_texts.append(text)

            # Process in batches
            print(f"Processing {len(cleaned_texts)} texts in batches of {batch_size}")
            for i in range(0, len(cleaned_texts), batch_size):
                batch = cleaned_texts[i:i + batch_size]
                try:
                    response = await self.client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=batch
                    )
                    all_embeddings.extend([e.embedding for e in response.data])
                    print(f"Processed batch {i//batch_size + 1}/{(len(cleaned_texts) + batch_size - 1)//batch_size}")
                    
                    # Add small delay between batches
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error in batch {i//batch_size + 1}: {str(e)}")
                    print(f"Batch size: {len(batch)}")
                    print(f"First text in failed batch: {batch[0][:200]}...")
                    raise

            return np.array(all_embeddings)
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise

    async def generate_cluster_summary(self, cluster_data):
        """Generate summary for a cluster"""
        system_prompt = """You are a geographic analyst. Your task is to name and describe urban areas based on their visual characteristics."""
        
        prompt = f"""Analyze these captions given, extract the key entities and relationships between them, and then create a summary of the group captions:

Location Descriptions:
{cluster_data}

Return your analysis in this exact JSON format:
{{
    "name": "3-5 word descriptive name",
    "summary": "2-3 sentences describing key features",
    "relationships": ["relationships"]
}}

Example Response:
{{
    "name": "Forested Campus Pathway Area",
    "summary": "A network of pedestrian pathways through a densely wooded campus setting. Large deciduous trees create natural archways over the paths.",
    "relationships": [
        "Paths connect building clusters",
        "Trees line walkways",
        "Green spaces border paths"
    ]
}}"""

        try:
            response = await self.generate_response(prompt, system_prompt)
            if not response:
                raise ValueError("Empty response from LLM")
            
            print(f"\nRaw LLM Response:\n{response}")
            return self.parse_summary_response(response)
            
        except Exception as e:
            print(f"Error in generate_cluster_summary: {str(e)}")
            raise

    async def process_batch_summaries(self, batch_requests):
        """Process a batch of summary requests"""
        batch_input_path = Path("batch_summaries.jsonl")
        try:
            # Save batch requests
            with open(batch_input_path, "w") as f:
                for request in batch_requests:
                    f.write(json.dumps(request) + "\n")
            
            # Create batch file
            batch_file = await self.client.files.create(
                file=open(batch_input_path, "rb"),
                purpose="batch"
            )
            
            # Create and monitor batch job
            batch = await self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            while True:
                batch_status = await self.client.batches.retrieve(batch.id)
                if batch_status.status in ['completed', 'failed', 'expired']:
                    break
                await asyncio.sleep(30)
            
            if batch_status.status == 'completed':
                output_file = await self.client.files.content(batch_status.output_file_id)
                return [json.loads(line) for line in output_file.text.split('\n') if line]
            
            return None
            
        finally:
            if batch_input_path.exists():
                batch_input_path.unlink()

    def parse_summary_response(self, content):
        """Parse JSON summary response"""
        try:
            # First remove any markdown code block formatting
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            
            # Find and parse the JSON object
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                print("No JSON object found in response")
                raise ValueError("Invalid response format")
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # Validate required fields exist
            required_fields = ['name', 'summary', 'relationships']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                print(f"Missing required fields: {missing_fields}")
                raise ValueError("Missing required fields")
            
            # Basic validation
            if not isinstance(data['name'], str) or not data['name'].strip():
                raise ValueError("Name cannot be empty")
            
            if not isinstance(data['summary'], str) or not data['summary'].strip():
                raise ValueError("Summary cannot be empty")
            
            if not isinstance(data['relationships'], list) or not data['relationships']:
                raise ValueError("Relationships must be a non-empty list")
            
            # Clean and return the data
            return {
                'name': data['name'].strip(),
                'summary': data['summary'].strip(),
                'relationships': [r.strip() for r in data['relationships'] if r.strip()]
            }
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Content received: {content[:200]}...")
            raise
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(f"Content received: {content[:200]}...")
            raise

    async def batch_generate_summaries(self, cluster_texts):
        """Generate summaries for multiple clusters"""
        summaries = []
        
        try:
            # Process each text individually
            for text in tqdm(cluster_texts, desc="Generating summaries"):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a geographic analyst. Return your analysis in JSON format."},
                            {"role": "user", "content": f"""Analyze these location descriptions and create a descriptive summary:

Location Descriptions:
{text}

Return your analysis as JSON:
{{
    "name": "descriptive name",
    "summary": "2-3 sentences describing key features",
    "relationships": ["relationship 1", "relationship 2", "relationship 3"]
}}"""}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    
                    content = response.choices[0].message.content
                    summary = self.parse_summary_response(content)
                    summaries.append(summary)
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    import traceback
                    print(f"Error generating summary: {str(e)}")
                    print(traceback.format_exc())
                    raise
            
            return summaries
            
        except Exception as e:
            import traceback
            print(f"Error processing batch: {str(e)}")
            print(traceback.format_exc())
            raise

    async def batch_generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """
        Generate embeddings for a large batch of texts efficiently
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embeddings as numpy arrays
        """
        try:
            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            print(f"\nGenerating embeddings for {len(texts)} texts in {total_batches} batches")
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Clean batch texts
                cleaned_batch = []
                for text in batch:
                    # Convert to string and basic cleaning
                    text = str(text).strip()
                    
                    # Remove markdown-style formatting
                    text = text.replace('**NAME:**', 'Name:')
                    text = text.replace('**DESCRIPTION:**', 'Description:')
                    text = text.replace('**', '')
                    
                    # Replace newlines with spaces and remove multiple spaces
                    text = ' '.join(text.split())
                    
                    # Handle empty strings
                    if not text:
                        text = "empty text"
                        
                    cleaned_batch.append(text)
                
                try:
                    # Generate embeddings for batch
                    response = await self.client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=cleaned_batch
                    )
                    
                    # Extract embeddings from response
                    batch_embeddings = [e.embedding for e in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    print(f"✓ Processed batch {i//batch_size + 1}/{total_batches}")
                    
                    # Add small delay between batches
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error in batch {i//batch_size + 1}: {str(e)}")
                    print(f"First text in failed batch: {cleaned_batch[0][:200]}...")
                    # Return empty embeddings for failed texts
                    all_embeddings.extend([np.zeros(1536) for _ in range(len(batch))])
            
            return all_embeddings
            
        except Exception as e:
            print(f"Error in batch_generate_embeddings: {str(e)}")
            # Return empty embeddings for all texts
            return [np.zeros(1536) for _ in range(len(texts))]
