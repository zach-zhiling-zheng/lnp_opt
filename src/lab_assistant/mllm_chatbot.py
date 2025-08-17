import os
import json
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from .helpers import (
    capture_image,
    monitor_equipment_status,
    web_search,
    generate_tecan_csv,
    get_current_datetime,
    image_to_base64,
)
from .audio_utils import record_audio, audio_to_text, text_to_audio, play_audio_with_skip
from .functions_definition import functions_definition

load_dotenv()

class MLLM_ChatBot:
    def __init__(self, api_key: str = None, model: str = "gpt-4o", helper_functions=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Put it in your environment or .env file.")
        self.model = model
        today = datetime.now()
        self.system_prompt = (
            "You are a laboratory assistant. Today's date is "
            f"{today.year}-{today.month:02d}-{today.day:02d}. "
            "You can capture and review images, monitor equipment folders, run web search, "
            "report the current datetime, and generate Tecan CSV pick lists."
        )
        self.helper_functions = helper_functions or []
        self.chat_history = [{"role": "system", "content": self.system_prompt}]

    def reset(self):
        self.chat_history = [{"role": "system", "content": self.system_prompt}]
        for f in os.listdir():
            if f.endswith(".mp3") or f.endswith(".wav"):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def _safe_extract_json(self, s: str):
        """Extract the first top-level JSON object from an LLM response string."""
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                return None
        return None

    def chat(self, user_input: str = None, output_mode: str = "text", max_tokens: int = 4096):
        client = OpenAI(api_key=self.api_key)

        if output_mode == "voice":
            self.chat_history[0]["content"] += (
                " Please answer in at most two sentences per reply. Offer details on request."
            )

        if user_input is None:
            audio_file_path = record_audio(10)
            user_input = audio_to_text(audio_file_path, self.api_key)

        self.chat_history.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=self.chat_history,
                tools=(self.helper_functions or functions_definition),
                tool_choice="auto",
                max_tokens=max_tokens,
            )
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return str(e)

        assistant_message = response.choices[0].message
        tool_calls = getattr(assistant_message, "tool_calls", None)

        if tool_calls:
            available = {
                "capture_image": capture_image,
                "monitor_equipment_status": monitor_equipment_status,
                "web_search": web_search,
                "generate_tecan_csv": generate_tecan_csv,
                "get_current_datetime": get_current_datetime,
            }

            for tc in tool_calls:
                fname = tc.function.name
                if fname in available:
                    args = json.loads(tc.function.arguments or "{}")
                    try:
                        res = available[fname](**args)
                    except Exception as e:
                        res = f"Tool error: {e}"

                    content = res if isinstance(res, str) else json.dumps(res)
                    tool_msg = {
                        "tool_call_id": tc.id,
                        "role": "function",
                        "name": fname,
                        "content": content,
                    }

                    # If the tool created or returned an image path, we provide a synthetic user message with the image data
                    if fname in ["capture_image", "monitor_equipment_status"] and isinstance(res, str) and os.path.exists(res):
                        # Replace tool content with a generic ack to keep tokens lower
                        tool_msg["content"] = "Image capture or search completed."
                        self.chat_history.append(tool_msg)

                        enc = image_to_base64(res)
                        self.chat_history.append({
                            "role": "assistant",
                            "content": "Image captured. Please analyze as needed."
                        })
                        self.chat_history.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Please examine this picture for context."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc}"}}
                            ]
                        })
                    else:
                        self.chat_history.append(tool_msg)
                else:
                    self.chat_history.append({
                        "tool_call_id": tc.id,
                        "role": "function",
                        "name": fname,
                        "content": "Unknown tool"
                    })

            # Final follow-up turn without more tool use
            response = client.chat.completions.create(
                model=self.model,
                messages=self.chat_history,
                tools=(self.helper_functions or functions_definition),
                tool_choice="none",
            )

        final_text = response.choices[0].message.content
        self.chat_history.append({"role": "assistant", "content": final_text})

        if output_mode == "voice":
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_mp3 = f"answer_{now}.mp3"
            text_to_audio(final_text, out_mp3, self.api_key)
            play_audio_with_skip(out_mp3)
        else:
            print(final_text)

        return final_text
