import os
import re
import json
import copy
import openai
from typing import Dict, Any, Optional, List
import numpy as np
from dotenv import load_dotenv; load_dotenv()
from together import Together
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

class MASEval:
    """
    Multi-Agent System Evaluation class for analyzing conversation logs
    and attributing failures in multi-agent systems.
    """
    
    def __init__(self, api_key: str = None, base_url: str = None, 
                 model_type: str = "openai", model_name: str = None, 
                 hf_token: str = None, device: str = "auto"):
        """
        Initialize MASEval with model configuration.
        
        Args:
            api_key (str): API key (for openai/together/claude model_type)
            base_url (str, optional): Custom OpenAI API base URL (ignored for together/claude)
            model_type (str): "openai", "together", "llama", "qwen", or "claude"
            model_name (str, optional): Specific model name to use
            hf_token (str, optional): Hugging Face token for private models
            device (str): Device for local models ("auto", "cpu", "cuda")
        """
        self.model_type = model_type
        self.temperature = 1.0
        self.max_tokens = 4196
        
        if model_type == "openai":
            if api_key:
                self.client = openai.OpenAI(api_key=api_key, base_url=base_url or "https://api.openai.com/v1")
            elif os.getenv("OPENAI_API_KEY"):
                self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url or "https://api.openai.com/v1")
            else:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

            self.model = model_name or "o3"
            
        elif model_type == "together":
            if os.getenv("TOGETHER_API_KEY"):
                self.client = Together()
            else:
                raise ValueError("TogetherAI API key not provided. Set TOGETHER_API_KEY environment variable or pass api_key parameter.")

            self.model = model_name or "openai/gpt-oss-120b"
            
        elif model_type == "claude":
            if Anthropic is None:
                raise ImportError("anthropic package is required for Claude models. "
                                "Install with: pip install anthropic")
            if api_key:
                self.client = Anthropic(api_key=api_key)
            elif os.getenv("ANTHROPIC_API_KEY"):
                self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            else:
                raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
            
            self.model = model_name or "claude-sonnet-4-5"
            
        elif model_type in ["llama", "qwen"]:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
            except ImportError:
                raise ImportError("transformers and torch are required for Hugging Face models. "
                                "Install with: pip install transformers torch")
            
            # Set default model names
            if model_type == "llama":
                self.model = model_name or "meta-llama/Llama-2-7b-chat-hf"
            elif model_type == "qwen":
                self.model = model_name or "Qwen/Qwen3-8B"
            
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model, 
                trust_remote_code=True,
                token=hf_token
            )
            
            # Set device
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                token=hf_token
            )
            
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'openai', 'together', 'llama', 'qwen', or 'claude'")
    
    def _chat_completion(self, messages: List[Dict[str, str]], model: str = None) -> str:
        """
        Generate chat completion using the configured model.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            str: Generated response
        """
            
        if self.model_type in ["openai", "together"]:
            create_args = dict(
                model=self.model if model is None else model,
                messages=messages,
                temperature=self.temperature,
            )

            # Determine if model requires a different max tokens parameter
            model_name_to_use = self.model if model is None else model
            # Support: all o3 models, or models requiring 'max_completion_tokens'
            if "o3" in model_name_to_use.lower() or 'gpt-5' in model_name_to_use.lower():
                create_args["max_completion_tokens"] = self.max_tokens
            else:
                create_args["max_tokens"] = self.max_tokens

            response = self.client.chat.completions.create(**create_args)
            return response
            
        elif self.model_type == "claude":
            # Claude API uses a different message format
            # Extract system message if present
            system_message = None
            claude_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                elif msg["role"] in ["user", "assistant"]:
                    claude_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            create_args = dict(
                model=self.model if model is None else model,
                messages=claude_messages,
                max_tokens=self.max_tokens,
            )
            
            if system_message:
                create_args["system"] = system_message
            
            if self.temperature > 0:
                create_args["temperature"] = self.temperature
            
            response = self.client.messages.create(**create_args)
            # Create a mock response object similar to OpenAI format for compatibility
            class ClaudeResponse:
                def __init__(self, content):
                    class Message:
                        def __init__(self, content):
                            self.content = content
                            self.reasoning = None
                    class Choice:
                        def __init__(self, content):
                            self.message = Message(content)
                    self.choices = [Choice(content)]
            
            return ClaudeResponse(response.content[0].text)
            
        elif self.model_type in ["llama", "qwen"]:
            import torch
            
            # Format messages for HuggingFace models
            if hasattr(self.tokenizer, 'apply_chat_template'):
                # Use chat template if available
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # Fallback to manual formatting
                formatted_prompt = self._format_messages_to_prompt(messages)
            
            # Tokenize
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            if torch.cuda.is_available() and self.device != "cpu":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=self.temperature > 0,
                    top_p=0.9 if self.temperature>0 else None,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
    
    def _format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string for models without chat template"""
        prompt_parts = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)
    
    def load_conversation_data(self, file_path: str) -> Dict[str, Any]:
        """
        Load conversation data from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file containing conversation data
            
        Returns:
            Dict[str, Any]: Parsed conversation data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception:
                f.seek(0)
                data = f.read()
        
        if not data.get("history"):
            raise ValueError(f"No chat history found in {file_path}")
            
        return data
    
    def format_chat_content(self, chat_history: List[Dict[str, Any]]) -> str:
        """
        Format chat history into a readable string.
        
        Args:
            chat_history (List[Dict]): List of chat entries
            
        Returns:
            str: Formatted chat content
        """
        return "\n\n".join([
            f"Step {entry.get('step')}: {entry.get('source') if entry.get('source') else (entry.get('role') if entry.get('role') else 'Unknown Agent')}: {entry.get('content', '')}"
            for entry in chat_history
        ])
    
    def evaluate_all_at_once(self, 
                            chat_history: List[Dict[str, Any]], 
                            problem: str, 
                            ground_truth: str) -> str:
        """
        Evaluate the conversation using the 'all at once' method.
        Analyzes the entire conversation to identify errors and responsible agents.
        
        Args:
            chat_history (List[Dict]): Chat conversation history
            problem (str): The problem statement
            ground_truth (str): The correct answer/solution
            
        Returns:
            str: Failure attribution analysis
        """
        chat_content = self.format_chat_content(chat_history)
   
        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real-world problem. "
            f"The problem is: {problem}\n"
            f"{f'The Answer for the problem is: {ground_truth}' if ground_truth != '' else ''}\n"
            "Identify all agents who made errors, at which steps each error occurred, and analyze the nature of these errors.\n\n"
            "Here's the conversation:\n"
            "--------------------------\n"
            f"{chat_content}\n"
            "--------------------------\n\n"
            "Based on this conversation, please provide a detailed analysis of the mistakes in the following JSON format. "
            "The output must be a list of objects, ordered by the step number. Each object must include:\n\n"
            "1. agent_name: The name of the agents who made any mistakes during the conversation. Directly output the name of the Expert.\n"
            "2. step_number: The integer step number where the mistake occurred (e.g., 0, 1, 2...).For example, in a conversation structured as follows: "
            """
            {
                "agent a": "xx",
                "agent b": "xxxx",
                "agent c": "xxxxx",
                "agent a": "xxxxxxx"
            },
            """

            "the textual chat history is structured as follows: "
            """
            Step 0: agent a: x
            Step 1: agent b: xxxx
            Step 2: agent c: xxxxx
            Step 3: agent a: xxxxxxx
            """

            "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. Please determine the step numbers where the mistakes occurred.\nPlease specify the step number as presented. \n"
            "3. failure_reason: Briefly describe the reason for failure in clear, natural language.\n"
            "4. ideal_action: Suggest the ideal action the agent should have taken. Do not include actual simulation results; "
            "instead, provide guidance aligned with the failure reason (e.g., 'The WebSurfer agent should have scrolled down the page to explore the information instead of visiting other pages').\n\n"
            "Please answer ONLY in the following JSON format:\n"
            '''[
                {
                    "agent_name": "AgentName1",
                    "step_number": "only integer",
                    "failure_reason": "Description of why it failed",
                    "ideal_action": "Guidance on what should have been done"
                },
                {
                    "agent_name": "AgentName2",
                    "step_number": "only integer",
                    "failure_reason": "Description of why it failed",
                    "ideal_action": "Guidance on what should have been done"
                },
                ...
            ]\n\n
            '''
            "Please order the results by the step number. Do not include any other text or comments."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
            {"role": "user", "content": prompt},
        ]
        
        retry_count = 0
        while True:
            try:
                message = self._chat_completion(messages)
                if 'oss' in self.model:
                    response = message.choices[0].message.content.strip()
                    reasoning = message.choices[0].message.reasoning.strip()
                elif 'R1' in self.model and 'together' == self.model_type:
                    response = message.choices[0].message.content.strip()
                    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
                    if think_match:
                        reasoning = think_match.group(1).strip()
                        # Response is everything after </think>
                        response = response[think_match.end():].strip()
                    else:
                        reasoning = ''
                elif self.model_type == 'qwen':
                    think_match = re.search(r"<think>(.*?)</think>", message, re.DOTALL)
                    if think_match:
                        reasoning = think_match.group(1).strip()
                        # Response is everything after </think>
                        response = message[think_match.end():].strip()
                    else:
                        reasoning = ''
                        response = message.strip()
                elif self.model_type == 'claude':
                    response = message.choices[0].message.content.strip()
                    reasoning = None
                else:
                    response = message.choices[0].message.content.strip()
                    reasoning = None
                json_result = json.loads(response.replace('```json', '').replace('```', '').strip())
                break
            except Exception as e:
                retry_count += 1
                print(e)
                print('Error occurred in failure_attribution, retrying...')
                if retry_count >= 10:
                    print('Stopping after 10 or more failures.')
                    raise RuntimeError('More than 10 errors occurred in failure_attribution.')
                continue
        return json_result, reasoning
    
    def evaluate_all_at_once_taxonomy(self, 
                            chat_history: List[Dict[str, Any]], 
                            problem: str, 
                            ground_truth: str) -> tuple:
        """
        Evaluate the conversation using the 'all at once' method with taxonomy classification.
        Analyzes the entire conversation to identify errors, responsible agents, and failure categories.
        
        Args:
            chat_history (List[Dict]): Chat conversation history
            problem (str): The problem statement
            ground_truth (str): The correct answer/solution
            
        Returns:
            tuple: (Failure attribution analysis with taxonomy, reasoning)
        """
        # Load taxonomy from file
        taxonomy_path = os.path.join(os.path.dirname(__file__), "taxonomy.txt")
        if not os.path.exists(taxonomy_path):
            # Fallback: try current directory
            taxonomy_path = "taxonomy.txt"
        
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            taxonomy_content = f.read()
        
        chat_content = self.format_chat_content(chat_history)
        
        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real-world problem. "
            f"The problem is: {problem}\n"
            f"{f'The Answer for the problem is: {ground_truth}' if ground_truth != '' else ''}\n"
            "Identify all agents who made errors, at which steps each error occurred, and analyze the nature of these errors.\n\n"
            "Here's the conversation:\n"
            "--------------------------\n"
            f"{chat_content}\n"
            "--------------------------\n\n"
            "Based on this conversation, please provide a detailed analysis of the mistakes in the following JSON format. "
            "The output must be a list of objects, ordered by the step number. Each object must include:\n\n"
            "1. agent_name: The name of the agents who made any mistakes during the conversation. Directly output the name of the Expert.\n"
            "2. step_number: The integer step number where the mistake occurred (e.g., 0, 1, 2...).For example, in a conversation structured as follows: "
            """
            {
                "agent a": "xx",
                "agent b": "xxxx",
                "agent c": "xxxxx",
                "agent a": "xxxxxxx"
            },
            """

            "the textual chat history is structured as follows: "
            """
            Step 0: agent a: x
            Step 1: agent b: xxxx
            Step 2: agent c: xxxxx
            Step 3: agent a: xxxxxxx
            """

            "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. Please determine the step numbers where the mistakes occurred.\nPlease specify the step number as presented. \n"
            "3. failure_reason: Briefly describe the reason for failure in clear, natural language.\n"
            "4. ideal_action: Suggest the ideal action the agent should have taken. Do not include actual simulation results; "
            "instead, provide guidance aligned with the failure reason (e.g., 'The WebSurfer agent should have scrolled down the page to explore the information instead of visiting other pages').\n"
            "5. fail_category: Classify the failure into one of the following categories based on the taxonomy below. "
            "You must select exactly one category that best matches the type of error:\n\n"
            f"{taxonomy_content}\n\n"
            "Please answer ONLY in the following JSON format:\n"
            '''[
                {
                    "agent_name": "AgentName1",
                    "step_number": "only integer",
                    "failure_reason": "Description of why it failed",
                    "ideal_action": "Guidance on what should have been done",
                    "fail_category": "Category name from the taxonomy above (e.g., "Planning Errors", "Verification Errors", etc.)"
                },
                {
                    "agent_name": "AgentName2",
                    "step_number": "only integer",
                    "failure_reason": "Description of why it failed",
                    "ideal_action": "Guidance on what should have been done",
                    "fail_category": "Category name from the taxonomy above (e.g., "Planning Errors", "Verification Errors", etc.)"
                },
                ...
            ]\n\n
            '''
            "Please order the results by the step number. Do not include any other text or comments."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
            {"role": "user", "content": prompt},
        ]
        
        retry_count = 0
        while True:
            try:
                message = self._chat_completion(messages)
                if 'oss' in self.model:
                    response = message.choices[0].message.content.strip()
                    reasoning = message.choices[0].message.reasoning.strip()
                elif 'R1' in self.model and 'together' == self.model_type:
                    response = message.choices[0].message.content.strip()
                    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
                    if think_match:
                        reasoning = think_match.group(1).strip()
                        # Response is everything after </think>
                        response = response[think_match.end():].strip()
                    else:
                        reasoning = ''
                elif self.model_type == 'qwen':
                    think_match = re.search(r"<think>(.*?)</think>", message, re.DOTALL)
                    if think_match:
                        reasoning = think_match.group(1).strip()
                        # Response is everything after </think>
                        response = message[think_match.end():].strip()
                    else:
                        reasoning = ''
                        response = message.strip()
                elif self.model_type == 'claude':
                    response = message.choices[0].message.content.strip()
                    reasoning = None
                else:
                    response = message.choices[0].message.content.strip()
                    reasoning = None
                json_result = json.loads(response.replace('```json', '').replace('```', '').strip())
                break
            except Exception as e:
                retry_count += 1
                print(e)
                print('Error occurred in failure_attribution, retrying...')
                if retry_count >= 10:
                    print('Stopping after 10 or more failures.')
                    raise RuntimeError('More than 10 errors occurred in failure_attribution.')
                continue
        return json_result, reasoning
    
    def evaluate_file(self, 
                     file_path: str, 
                     method: str = "all_at_once", 
                     output_path: Optional[str] = None, 
                     gt: bool = False) -> Dict[str, Any]:
        """
        Evaluate a single conversation file using the specified method.
        
        Args:
            file_path (str): Path to the conversation JSON file
            method (str): Evaluation method ("all_at_once" or "subtask")
            output_path (str, optional): Path to save results
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        # Load data
        data = self.load_conversation_data(file_path)
        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "") if gt else ""
        
        # Create result data
        result_data = copy.deepcopy(data)
        
        # Evaluate based on method
        if method == "all_at_once":
            failure_attribution = self.evaluate_all_at_once(chat_history, problem, ground_truth)
            result_data["failure_attribution"], result_data["failure_attribution_reasoning"] = failure_attribution
        elif method == "all_at_once_taxonomy":
            failure_attribution = self.evaluate_all_at_once_taxonomy(chat_history, problem, ground_truth)
            result_data["failure_attribution"], result_data["failure_attribution_reasoning"] = failure_attribution
        else:
            raise ValueError(f"Unknown method: {method}.")
        
        # Save results if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=4)
        
        return result_data