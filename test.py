import time
import ollama

start = time.perf_counter()          # start the stopwatch

stream = ollama.chat(
    model="qwen2.5:3b",
    messages=[{"role": "system", "content": "You answer in exactly one sentence."},
                {"role": "user", "content": "Hello, how are you?"}],
    stream=True,                     # get the reply piece by piece
)

first_token_time = None

for chunk in stream:                 # each chunk holds a few characters/words
    if first_token_time is None:
        first_token_time = time.perf_counter() - start
    print(chunk.message.content, end="", flush=True)

total_time = time.perf_counter() - start

print()                              # finish the line after the reply
print(f"First words after: {first_token_time:.2f}s")
print(f"Total time:        {total_time:.2f}s")
