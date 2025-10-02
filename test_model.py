import os
import models


"""
# model = models.OpenAIModel(model_name="gpt-3.5-turbo")
# model = models.GeminiPro(model_name="gemini-pro")
# model = models.Llama(model_name="llama2-7b-chat")

model_name:
"gpt-3.5-turbo", "gpt-4", "gemini-pro", "llama2-7b-chat", "llama2-13b-chat", "phi-2", "Meta-Llama-3-8B-Instruct"
"codellama-7b-instruct", "mistral-7b-instruct-v0.2", "codeqwen1.5-7b-chat",
"granite-3b-code-instruct", "granite-8b-code-instruct", "granite-20b-code-instruct"
"""
model_name = "codellama-70b-instruct"
print("\nSuppport model list:\n", models.LLM.model_list())
print("\nModel name:", model_name)
model = models.LLM.create_model(model_name=model_name)


for _ in range(10):
    question = input("\033[33m" + "LLM> " + "\033[m")
    # question = "Please takeoff and fly to [10, 10, 10]."

    if question == "!quit" or question == "!exit":
        break

    if question == "!clear":
        os.system("cls")
        continue

    response = model.predict(prompt=question)

    print(f"\n{response}\n")


