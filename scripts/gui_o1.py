import tkinter as tk
from tkinter import scrolledtext, ttk
import time
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_openai(event=None):
    user_input = input_box.get("1.0", tk.END).strip()
    selected_model = model_var.get()
    start = time.time()
    resp = CLIENT.chat.completions.create(
        model=selected_model,
        messages=[{"role": "user", "content": [{"type": "text", "text": user_input}]}],
    )
    answer = resp.choices[0].message.content
    took = time.time() - start
    response_time_label.config(text=f"Response time: {took:.2f} seconds")
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, answer)

def adjust_font_size(event=None):
    size = int(font_size_scrollbar.get())
    input_box.config(font=("Arial", size))
    output_box.config(font=("Arial", size))

root = tk.Tk()
root.title("OpenAI Chat GUI")
root.state("zoomed")

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=2)
root.grid_rowconfigure(0, weight=1)

left_frame = tk.Frame(root, bd=2, relief=tk.RAISED)
left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

right_frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

input_box_label = tk.Label(left_frame, text="Enter your question:")
input_box_label.pack(pady=10)

input_box = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, height=10, width=50, font=("Arial", 12))
input_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
input_box.bind("<Shift-Return>", ask_openai)

model_var = tk.StringVar(value="o1-mini")
model_label = tk.Label(left_frame, text="Select Model:")
model_label.pack(pady=10)
model_dropdown = tk.OptionMenu(left_frame, model_var, "o1-mini", "o1-preview", "gpt-4o")
model_dropdown.pack(pady=10)

submit_button = tk.Button(left_frame, text="Submit", command=ask_openai)
submit_button.pack(pady=10)

response_time_label = tk.Label(left_frame, text="Response time: ")
response_time_label.pack(pady=10)

output_box_label = tk.Label(right_frame, text="Response:")
output_box_label.pack(pady=10)
output_box = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, font=("Arial", 12))
output_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

font_size_label = tk.Label(left_frame, text="Adjust font size:")
font_size_label.pack(pady=10)

font_size_scrollbar = ttk.Scale(left_frame, from_=8, to_=24, orient="horizontal", command=adjust_font_size)
font_size_scrollbar.set(12)
font_size_scrollbar.pack(pady=10)

root.mainloop()
