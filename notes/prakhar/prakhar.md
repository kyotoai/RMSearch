# vllm with GPT oss

The model files are already downlaoded and just run 
vllm serve ./gpt-oss-20b     --host 0.0.0.0     --port 7000 

from workspace, this will start the vllm server with GPT oss

here is s sample python inference code with API

```python
from openai import OpenAI
 
client = OpenAI(
    base_url="http://localhost:7000/v1",
    api_key="EMPTY"
)
 
result = client.chat.completions.create(
    model="./gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what data vs task parallelism is."}
    ]
)
 
print(result.choices[0].message.content)
 ```