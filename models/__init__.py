from .models import OpenAI, GeminiPro, Llama, Phi, Mistral, Qwen, Llama3, Granite
from .chat_history import ChatHistory

# 'llama2-7b', 'llama2-7b-chat', 'llama2-13b', 'llama2-13b-chat', 'llama2-70b', 'llama2-70b-chat'
# 'codellama-7b', 'codellama-7b-instruct', 'codellama-7b-python'
# 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106'
MODEL_LIST = {
    "Llama": ['llama2-7b-chat', 'codellama-7b-instruct', 'codellama-13b-instruct', 'codellama-34b-instruct', 'codellama-70b-instruct'],
    "Llama3": ['meta-llama-3-8b-instruct'],
    "OpenAI": ['gpt-3.5-turbo', 'gpt-4'],
    "GeminiPro": ['gemini-pro'],
    "Phi": ['phi-1.5', 'phi-2'],
    "Mistral": ['mistral-7b-instruct-v0.2'],
    "Qwen": ['codeqwen1.5-7b-chat'],
    "Granite" : ['granite-3b-code-instruct', 'granite-8b-code-instruct', 'granite-20b-code-instruct']
}

SUPPORTED_MODELS = [model for model_class in MODEL_LIST.keys() for model in MODEL_LIST[model_class]]


class LLM:
    @staticmethod
    def model_list():
        return SUPPORTED_MODELS

    @staticmethod
    def create_model(model_name):
        for model_class, model_names in MODEL_LIST.items():
            if model_name.lower() in model_names:
                return globals()[model_class](model_name=model_name)
        raise ValueError("The model is not supported!")
