import argparse
from src.lab_assistant.mllm_chatbot import MLLM_ChatBot
from src.lab_assistant.functions_definition import functions_definition

def main(model: str):
    bot = MLLM_ChatBot(model=model, helper_functions=functions_definition)
    print("Chat is ready. Type your message and press Enter. Ctrl+C to exit.")
    while True:
        try:
            user = input("> ").strip()
            if not user:
                continue
            bot.chat(user_input=user, output_mode="text")
        except KeyboardInterrupt:
            print("\nBye.")
            break

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="gpt-4o")
    args = p.parse_args()
    main(args.model)
