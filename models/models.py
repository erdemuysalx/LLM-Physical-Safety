import json
import os
from abc import ABC

import torch
import openai
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM

from .chat_history import ChatHistory


def get_config(key):
    file_path = os.path.abspath(__file__)
    parent_dir = os.path.join(os.path.dirname(file_path), '..')
    file_path = os.path.join(parent_dir, 'config.json')
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config[key]


class Config:
    temperature = 0
    max_new_tokens = 512


class OpenAI:
    """
    Large language model for interfacing with OpenAI's GPT models.

    Parameters:
    -----------
    model : str
        The name of the OpenAI model.
    max_new_tokens : int
        The maximum number of new tokens to be generated (default is 256).
    temperature : float
        The temperature for text generation (default is 0).
    system_prompt : str
        The system prompt to be used (default is None).

    Methods:
    --------
    predict(prompt)
        Predicts the output based on the given input text using the OpenAI model.
    """

    def __init__(self, model_name="gpt-3.5-turbo",
                 max_new_tokens=Config.max_new_tokens,
                 temperature=Config.temperature,
                 system_prompt=None,
                 no_example=False):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.chat_history = ChatHistory(system_prompt=self.system_prompt).openai
        if no_example:
            self.chat_history = self.chat_history[:1]
        self.client = openai.OpenAI(api_key=get_config("OPENAI_API_KEY"))

    def predict(self, prompt):
        messages = self.chat_history[:]
        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature
        )

        return completion.choices[0].message.content


class GeminiPro:
    """
    Large language model for interfacing with Gemini Pro models.

    Parameters:
    -----------
    model : str
        The name of the OpenAI model.
    max_new_tokens : int
        The maximum number of new tokens to be generated (default is 256).
    temperature : float
        The temperature for text generation (default is 0).
    system_prompt : str
        The system prompt to be used (default is None).

    Methods:
    --------
    predict(prompt)
        Predicts the output based on the given input text using the OpenAI model.
    """

    def __init__(self, model_name="gemini-pro",
                 max_new_tokens=Config.max_new_tokens,
                 temperature=Config.temperature,
                 system_prompt=None):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.chat_history = ChatHistory(system_prompt=self.system_prompt).gemini
        # Set up the model
        self.generation_config = {
            "temperature": self.temperature,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": self.max_new_tokens,
        }
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        genai.configure(api_key=get_config("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name=self.model_name,
                                           generation_config=self.generation_config,
                                           safety_settings=self.safety_settings)

    def predict(self, prompt):
        chat = self.model.start_chat(history=self.chat_history)
        try:
            response = chat.send_message(prompt)
            return response.text
        except Exception as error:
            return error


class LLMModel(ABC):
    def __init__(self, model_name, max_new_tokens, temperature, device):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

    def predict(self, prompt):
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        outputs = self.model.generate(input_ids,
                                      max_new_tokens=self.max_new_tokens,
                                      temperature=self.temperature)

        out = self.tokenizer.decode(outputs[0])
        return out


class Llama(LLMModel):
    """
    Large language model for interfacing with Llama 2 models.
    The temperature parameter also plays an important role for exploration, as a higher temperature enables us to sample more diverse outputs.
    """
    def __init__(self, model_name,
                 max_new_tokens=Config.max_new_tokens,
                 temperature=Config.temperature,
                 device='auto',
                 dtype='auto',
                 system_prompt=None,
                 model_dir=None):
        super(Llama, self).__init__(model_name, max_new_tokens, temperature, device)
        self.chat_history = ChatHistory(system_prompt=system_prompt)

        if model_dir is None:
            parts = model_name.lower().split('-')
            number = parts[1]
            is_chat = 'chat' in parts
            parent_dir = get_config("MODEL_DIR")

            if 'codellama' in model_name.lower():
                if 'instruct' in model_name:
                    model_dir = f"{parent_dir}/codellama/CodeLlama-{number}-Instruct-hf"
                elif 'python' in model_name:
                    model_dir = f"{parent_dir}/codellama/CodeLlama-{number}-Python-hf"
                else:
                    model_dir = f"{parent_dir}/codellama/CodeLlama-{number}-hf"
            else:
                model_dir = f"{parent_dir}/meta-llama/Llama-2-{number}"
                if is_chat:
                    model_dir += "-chat"
                model_dir += "-hf"

        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map=self.device, torch_dtype=dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=self.device, torch_dtype=dtype)

    def predict(self, prompt):
        input_text = f"<s>[INST] <<SYS>>{self.chat_history.system_prompt}<</SYS>>\n" \
                     f"{self.chat_history.q1} [/INST] {self.chat_history.a1} </s>\n" \
                     f"<s>[INST] {prompt} [/INST]"

        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        if self.temperature == 0:
            self.temperature += 0.0001

        outputs = self.model.generate(input_ids,
                                      max_new_tokens=self.max_new_tokens,
                                      temperature=self.temperature)

        out = self.tokenizer.decode(outputs[0],
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
        return out[len(input_text)-6:]


class Phi(LLMModel):
    """
    Language model class for the Phi model.

    Inherits from LMMBaseModel and sets up the Phi language model for use.

    Parameters:
    -----------
    model : str
        The name of the Phi model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str
        The dtype to use for inference (default is 'auto').
    """

    def __init__(self, model_name,
                 max_new_tokens=Config.max_new_tokens,
                 temperature=Config.temperature,
                 device='auto',
                 dtype='auto',
                 system_prompt=None,
                 model_dir=None
                 ):
        super(Phi, self).__init__(model_name, max_new_tokens, temperature, device)
        if model_dir is None:
            parent_dir = get_config("MODEL_DIR")
            model_dir = parent_dir + "/microsoft/phi-1_5" if model_name == "phi-1.5" else parent_dir + "/microsoft/phi-2"

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=dtype,
                                                       device_map=device)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=dtype,
                                                          device_map=device)
        self.chat_history = ChatHistory(system_prompt=system_prompt)

    def predict(self, prompt):
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device

        input_text = f"Human: {self.chat_history.system_prompt}\n" \
                     f"AI: No problem." \
                     f"Human: {self.chat_history.q1}" \
                     f"AI: {self.chat_history.a1}" \
                     f"Human: {prompt}" \
                     f"AI:"

        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        outputs = self.model.generate(input_ids,
                                      max_new_tokens=self.max_new_tokens,
                                      temperature=self.temperature+0.1)

        out = self.tokenizer.decode(outputs[0])
        return out[len(input_text):]


class Mistral(LLMModel):
    """
    Large language model for interfacing with Mistral-7B models.
    The temperature parameter also plays an important role for exploration, as a higher temperature enables us to sample more diverse outputs.
    """
    def __init__(self, model_name,
                 max_new_tokens=Config.max_new_tokens,
                 temperature=Config.temperature,
                 device='auto',
                 dtype='auto',
                 system_prompt=None,
                 model_dir=None):
        super(Mistral, self).__init__(model_name, max_new_tokens, temperature, device)

        if model_dir is None:
            if model_name.lower() == "mistral-7b-instruct-v0.2":
                parent_dir = get_config("MODEL_DIR")
                model_dir = f"{parent_dir}/mistralai/Mistral-7B-Instruct-v0.2"
            else:
                raise ValueError("Invalid model name.")

        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.chat_history = ChatHistory(system_prompt=system_prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map=self.device, torch_dtype=dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=self.device, torch_dtype=dtype)

    def predict(self, prompt):
        messages = [
            {"role": "user", "content": self.chat_history.system_prompt},
            {"role": "assistant", "content": "No problem."},
            {"role": "user", "content": self.chat_history.q1},
            {"role": "assistant", "content": self.chat_history.a1},
            {"role": "user", "content": prompt}
        ]
        encoded = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encoded.to(self.device)
        generated_ids = self.model.generate(model_inputs,
                                            max_new_tokens=Config.max_new_tokens,
                                            temperature=Config.temperature)
        input_token_len = encoded.shape[-1]
        decoded = self.tokenizer.batch_decode(generated_ids[:, input_token_len:])
        return decoded[0]


class Qwen(LLMModel):
    """
        Large language model for interfacing with CodeQwen1.5-7B-Chat models.
        The temperature parameter also plays an important role for exploration, as a higher temperature enables us to sample more diverse outputs.
    """
    def __init__(self, model_name,
                 max_new_tokens=Config.max_new_tokens,
                 temperature=Config.temperature,
                 device='auto',
                 dtype='auto',
                 system_prompt=None,
                 model_dir=None):
        super(Qwen, self).__init__(model_name, max_new_tokens, temperature, device)

        if model_dir is None:
            if model_name.lower() == "codeqwen1.5-7b-chat":
                parent_dir = get_config("MODEL_DIR")
                model_dir = f"{parent_dir}/Qwen/CodeQwen1.5-7B-Chat"
            else:
                raise ValueError("Invalid model name.")

        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.temperature = Config.temperature
        if self.temperature == 0:
            self.temperature += 0.0001
        self.chat_history = ChatHistory(system_prompt=system_prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map=self.device, torch_dtype=dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=self.device, torch_dtype=dtype)

    def predict(self, prompt):
        messages = [
            {"role": "system", "content": self.chat_history.system_prompt},
            {"role": "user", "content": self.chat_history.q1},
            {"role": "assistant", "content": self.chat_history.a1},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=Config.max_new_tokens,
            temperature=self.temperature
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        out = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return out

class Llama3(LLMModel):
    """
    Large language model for interfacing with Llama 3 models.
    """
    def __init__(self, model_name,
                 max_new_tokens=Config.max_new_tokens,
                 temperature=Config.temperature,
                 device='auto',
                 dtype='auto',
                 system_prompt=None,
                 model_dir=None):
        super(Llama3, self).__init__(model_name, max_new_tokens, temperature, device)
        self.chat_history = ChatHistory(system_prompt=system_prompt)

        if model_dir is None:
            parent_dir = get_config("MODEL_DIR")
            if model_name.lower() == 'meta-llama-3-8b-instruct':
                model_dir = f"{parent_dir}/meta-llama/Meta-Llama-3-8B-Instruct"

        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map=self.device, torch_dtype=dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=self.device, torch_dtype=dtype)

    def predict(self, prompt):
        messages = [
            {"role": "system", "content": self.chat_history.system_prompt},
            {"role": "user", "content": self.chat_history.q1},
            {"role": "assistant", "content": self.chat_history.a1},
            {"role": "user", "content": prompt}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=Config.max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        out = self.tokenizer.decode(response, skip_special_tokens=True)
        return out


class Granite(LLMModel):
    """
        Large language model for interfacing with IBM Granite Code.
    """
    def __init__(self, model_name,
                 max_new_tokens=Config.max_new_tokens,
                 temperature=Config.temperature,
                 device='auto',
                 dtype='auto',
                 system_prompt=None,
                 model_dir=None):
        super(Granite, self).__init__(model_name, max_new_tokens, temperature, device)
        if model_dir is None:
            if model_name.lower() == "granite-3b-code-instruct":
                parent_dir = get_config("MODEL_DIR")
                model_dir = f"{parent_dir}/ibm-granite/granite-3b-code-instruct"
            elif model_name.lower() == "granite-8b-code-instruct":
                parent_dir = get_config("MODEL_DIR")
                model_dir = f"{parent_dir}/ibm-granite/granite-8b-code-instruct"
            elif model_name.lower() == "granite-20b-code-instruct":
                parent_dir = get_config("MODEL_DIR")
                model_dir = f"{parent_dir}/ibm-granite/granite-20b-code-instruct"
            else:
                raise ValueError("Invalid model name.")

        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.chat_history = ChatHistory(system_prompt=system_prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map=self.device, torch_dtype=dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=self.device, torch_dtype=dtype)
        self.model.eval()

    def predict(self, prompt):
        chat = [
            {"role": "system", "content": self.chat_history.system_prompt},
            {"role": "user", "content": self.chat_history.q1},
            {"role": "assistant", "content": self.chat_history.a1},
            {"role": "user", "content": prompt}
        ]
        chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # tokenize the text
        input_tokens = self.tokenizer(chat, return_tensors="pt")
        # transfer tokenized inputs to the device
        for i in input_tokens:
            input_tokens[i] = input_tokens[i].to(self.device)
        # generate output tokens
        out = self.model.generate(**input_tokens, max_new_tokens=Config.max_new_tokens, temperature=Config.temperature)
        # decode output tokens into text
        out = self.tokenizer.batch_decode(out)[0]
        answer_index = out.rfind('Answer:\n')
        out = out[answer_index + len('Answer:\n'):].strip()
        return out