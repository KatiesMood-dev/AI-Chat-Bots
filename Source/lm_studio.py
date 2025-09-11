import time
import threading
import json
import lmstudio as lms

# === CONFIG ===
OUTPUT_FILE = "D:/My Projects/AI-Chat-Bots/Source/STT_Log.txt"   # Path to file with lines
CLIENT_CONFIG_FILE = "D:/My Projects/AI-Chat-Bots/clients.json"  # JSON file containing client info
CONTEXT_FILE = "D:/My Projects/AI-Chat-Bots/context.txt"  # Common background context for all clients

# === STATE ===
MODELS = {}
last_line_sent = {}
pausetimes = {}
last_request_time = {}
common_context = ""

# === FUNCTIONS ===
def read_lines(filepath):
    """Read all lines from file, stripped."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def parse_response(response: str):
    """Parse response of form '"some text" 5' → (string, int)."""
    try:
        # Find first and last quote
        first_quote = response.find('"')
        last_quote = response.rfind('"')
        if first_quote == -1 or last_quote == -1 or first_quote == last_quote:
            raise ValueError("Malformed response, missing quotes")

        text = response[first_quote + 1:last_quote]
        number_part = response[last_quote + 1:].strip()
        pause = int(number_part)
        return text, pause
    except Exception as e:
        print(f"[ERROR] Could not parse response: {response} ({e})")
        return response, 1  # fallback: 1 second pause

def client_worker(twitch_name):
    global last_line_sent, pausetimes, last_request_time
    model = MODELS[twitch_name]

    while True:
        now = time.time()
        # Check if client is allowed to send request
        if now - last_request_time[twitch_name] < pausetimes[twitch_name]:
            time.sleep(0.5)
            continue

        lines = read_lines(OUTPUT_FILE)
        next_line_index = last_line_sent[twitch_name] + 1

        if next_line_index < len(lines):
            message = lines[next_line_index]

            # === API CALL ===
            try:
                output = model.respond(message)
            except Exception as e:
                print(f"[ERROR] API request failed for client: {e}")
                time.sleep(2)
                continue

            # === PARSE RESPONSE ===
            print(f"Client {twitch_name} Response: {output.content}")
            _, pause = parse_response(output.content)

            # Update state
            last_line_sent[twitch_name] = next_line_index
            pausetimes[twitch_name] = pause
            last_request_time[twitch_name] = now
        else:
            # No new lines → wait a bit
            time.sleep(1)

def load_models_from_json(filepath):
    global MODELS, last_line_sent, pausetimes, last_request_time, common_context

    # Load common context
    with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
        common_context = f.read().strip()

    # Load client configurations
    with open(filepath, "r", encoding="utf-8") as f:
        configs = json.load(f)

    for cfg in configs:
        client = lms.Client("localhost:1234")
        model = client.llm.model("openai/gpt-oss-20b")

        twitch_name = cfg.get("name", f"client_{len(MODELS)}")
        # Prepend common context to personality
        personality = cfg.get("personality", "")
        client_personality = f"{common_context}\n\n{personality}" if common_context else personality
        print(f"sending to {twitch_name}: {client_personality}")
        response = model.respond(client_personality)
        print(f"{twitch_name}: {response}")

        MODELS[twitch_name] = model
        last_line_sent[twitch_name] = -1
        pausetimes[twitch_name] = 0
        last_request_time[twitch_name] = 0

def main():
    load_models_from_json(CLIENT_CONFIG_FILE)

    threads = []
    for twitch_name in MODELS:
        t = threading.Thread(target=client_worker, args=(twitch_name,), daemon=True)
        t.start()
        threads.append(t)

    # Keep main thread alive
    while True:
        time.sleep(5)

if __name__ == "__main__":
    main()
