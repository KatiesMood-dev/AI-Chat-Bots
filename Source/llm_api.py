import os
import json
import time
import random
from pathlib import Path
import lmstudio as lms
import tiktoken
from pydantic import BaseModel

# -----------------------
# CONFIG
# -----------------------
dir_path = os.path.dirname(os.path.realpath(__file__))
MIC_TRANSCRIPT_FILE = os.path.join(dir_path, "mic_transcript.txt")
COMMON_CONTEXT_FILE = os.path.join(dir_path, "common_context.txt")
PERSONALITIES_FILE = os.path.join(dir_path, "personalities.json")
LMS_SERVER_API_HOST = "localhost:1234"
LLM_MODEL = "openai/gpt-oss-20b"
TOKEN_THRESHOLD = 4000
# lmstudio client (make sure LM Studio server is running)
lms.configure_default_client(LMS_SERVER_API_HOST)

enc = tiktoken.get_encoding("o200k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

class TwitchChatResponse(BaseModel):
    twitch_chat_message: str
    delay_seconds: int

# -----------------------
# Load resources
# -----------------------
with open(COMMON_CONTEXT_FILE, "r", encoding="utf-8") as f:
    common_context = f.read().strip()

with open(PERSONALITIES_FILE, "r", encoding="utf-8") as f:
    personalities = json.load(f)


def get_recent_transcript(chat: lms.Chat, next_context_line : int):
    """Read last n lines from mic transcript file."""
    path = Path(MIC_TRANSCRIPT_FILE)
    if not path.exists():
        print("No transcript file found.")
        return next_context_line
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    len_lines = len(lines)
    transcript = ""
    if len_lines > next_context_line:
        transcript = "".join(lines[next_context_line:]).strip()
        next_context_line = len_lines
    print(f"[TRANSCRIPT] {transcript}")

    if len(transcript) > 0:
        message = f"The streamer just said (transcript of microphone): {transcript}" 
        chat.add_user_message(message)
    else:
        message = f"The streamer did not say anything since the last message."
        chat.add_user_message(message)
    return next_context_line



def build_context(personality):
    """Build system + user instructions for LLM."""
    return f"""{common_context}

Your Twitch name: {personality['name']}
Your personality: {personality['personality']}
"""

# The streamer just said (transcript of microphone):
# {transcript}
# 
# Now respond following the rules.

def reduce_chat_context(chat: lms.Chat, p, next_context_line) -> lms.Chat:
    messages = chat._messages
    contents = ""
    for message in messages:
        for content in message.content:
            contents += ''.join(content.text)
    tokenlist = enc.encode(contents)
    if len(tokenlist) > TOKEN_THRESHOLD:
        path = Path(MIC_TRANSCRIPT_FILE)
        if not path.exists():
            print("No transcript file found.")
            return chat
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
       
        len_lines = len(lines)
        transcript = ""
        if len_lines > next_context_line:
            transcript = "".join(lines[next_context_line-6:next_context_line-1]).strip()
            next_context_line = len_lines
        print(f"[TRANSCRIPT REBUILD] {transcript}")

        chat = lms.Chat(build_context(p))
        if len(transcript) > 0:
            message = f"The streamer just said (transcript of microphone): {transcript}" 
            chat.add_user_message(message)
        else:
            message = f"The streamer did not say anything yet."
            chat.add_user_message(message)

    return chat    


def generate_response(chat: lms.Chat, model: lms.LLM):
    # Generate a Twitch-style message for a given personality
    response = model.respond(chat, response_format=TwitchChatResponse)
    results = json.loads(response.content)
    # print(json.dumps(results, indent=2))
    return results


def main():
    print("Starting Twitch chat generator (lmstudio)...")
    chats: dict[str, lms.Chat] = {}
    next_message_time: dict[str, int] = {}
    next_context_line: dict[str, int] = {}
    model = lms.llm(LLM_MODEL)

    for p in personalities:
        chats[p["name"]] = lms.Chat(build_context(p))
        next_message_time[p["name"]] = 0
        next_context_line[p["name"]] = 0

    while True:
        now = time.time()
        for p in personalities:
            if now >= next_message_time[p["name"]]:
                next_context_line[p["name"]] = get_recent_transcript(chats[p["name"]], next_context_line[p["name"]])
                chats[p["name"]] = reduce_chat_context(chats[p["name"]], p, next_context_line[p["name"]])
                response = generate_response(chats[p["name"]], model)
                # Parse output
                try:
                    message_part = response["twitch_chat_message"]
                    delay_part = response["delay_seconds"]
                    print(f"[{p['name']}] {message_part} (next in {delay_part}s)")
                    next_message_time[p["name"]] = now + delay_part
                except Exception:
                    print(f"[{p['name']}] ⚠️ Unexpected format: {response}")
                    next_message_time[p["name"]] = now + random.randint(5, 15)

        time.sleep(1)


if __name__ == "__main__":
    main()
