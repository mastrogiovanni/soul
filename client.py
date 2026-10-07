from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-inference-key")
stream = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Hello!"}],
    # max_tokens=64,
    stream=True,
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
