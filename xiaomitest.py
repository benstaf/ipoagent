from openai import OpenAI

client = OpenAI(
    api_key="sk-sim8gcwob6f3wxdhcgzqo9edvxurqmtutzxe7084vo00g0im",
    base_url="https://api.xiaomimimo.com/v1"
)

print("Sending request...")

response = client.chat.completions.create(
    model="mimo-v2.5-pro",
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)

print("Raw response:")
print(response)

print("Message:")
print(response.choices[0].message.content)
