import google.generativeai as genai

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBYDWMvaZZkCljM1CuhLhnzcH4aUrsoQUk"
genai.configure(api_key=GEMINI_API_KEY)

# List available models
models = genai.list_models()
print("Available models:")
for model in models:
    print(model)