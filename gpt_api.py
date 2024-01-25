from openai import OpenAI

# api_key = "string"

client = OpenAI(api_key = "string")

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  max_tokens=200,
  n = 1,
  messages=[
    {"role": "user", "content": "What's the largest planet in the solar system"}
  ]
)

print(completion.choices[0].message)
