import os
from dotenv import load_dotenv
from openai import OpenAI

print("=" * 80)
print("Before load_dotenv")
print("OPENAI_BASE_URL =", repr(os.getenv("OPENAI_BASE_URL")))
print("OPENAI_API_KEY exists =", bool(os.getenv("OPENAI_API_KEY")))
print("=" * 80)

load_dotenv(override=True)

print("=" * 80)
print("After load_dotenv")
print("TESTED_MODEL =", repr(os.getenv("TESTED_MODEL")))
print("TESTED_MODEL_URL =", repr(os.getenv("TESTED_MODEL_URL")))

tested_key = os.getenv("TESTED_API_KEY")
print("TESTED_API_KEY exists =", bool(tested_key))

if tested_key:
    print("TESTED_API_KEY prefix =", tested_key[:12])
    print("TESTED_API_KEY length =", len(tested_key))
print("=" * 80)

# Reproduce get_agent.py logic
if os.getenv("TESTED_MODEL_URL"):
    os.environ["OPENAI_BASE_URL"] = os.getenv("TESTED_MODEL_URL")

if os.getenv("TESTED_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("TESTED_API_KEY")

print("=" * 80)
print("Effective values")
print("OPENAI_BASE_URL =", repr(os.getenv("OPENAI_BASE_URL")))

api_key = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY exists =", bool(api_key))

if api_key:
    print("OPENAI_API_KEY prefix =", api_key[:12])
    print("OPENAI_API_KEY length =", len(api_key))
print("=" * 80)

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)

print("Client created")
print("client.base_url =", client.base_url)

try:
    response = client.chat.completions.create(
        model="mimo-v2.5-pro",
        messages=[
            {"role": "user", "content": "hello"}
        ],
    )

    print("\nSUCCESS")
    print(response.choices[0].message.content)

except Exception as e:
    print("\nFAILED")
    print(type(e).__name__)
    print(e)
