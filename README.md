# Lab Automation Suite

A small, ready-to-run toolkit for:
- Capturing and inspecting camera images from your lab environment
- Monitoring equipment images stored by timestamped filenames
- Running a multimodal lab assistant that can call helper functions through OpenAI tools
- Generating Tecan CSVs for combinatorial dispense plans
- Benchmarking experiment-suggestion strategies on an LNP dataset
- Visualizing results and a quick Tkinter GUI to query OpenAI models

## Repo layout

```
lab-automation-suite/
├─ src/
│  ├─ lab_assistant/
│  │  ├─ helpers.py
│  │  ├─ audio_utils.py
│  │  ├─ mllm_chatbot.py
│  │  └─ functions_definition.py
│  └─ optimization/
│     ├─ lnp_optimizer.py
│     ├─ viz_pair_scatter.py
│     └─ viz_heatmap.py
├─ scripts/
│  ├─ run_chat.py
│  ├─ run_optimizer.py
│  └─ gui_o1.py
├─ data/
│  └─ README.md
├─ .env.example
├─ requirements.txt
├─ LICENSE
└─ README.md
```

## Quickstart

1. Create and activate a virtual environment.
2. Install deps:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Optional: set `CAMERA_URL`.
4. Run the chat assistant:

```bash
python -m scripts.run_chat --model gpt-4o
```

5. Run the LNP optimizer example with your `LNP.xlsx` in the repo root or `data/`:

```bash
python -m scripts.run_optimizer --n 4 --iterations 5 --repeat 2 --methods Random,BO,EDBO,LLM,R-LLM --xlsx LNP.xlsx
```

6. Launch the simple Tkinter GUI:

```bash
python -m scripts.gui_o1
```

## Notes

- `pyaudio` may require system packages for PortAudio.
- The `keyboard` package may need admin privileges to detect global keypresses. If it fails, the code falls back where possible.
- The GUI uses `o1-mini` and `o1-preview` model names by default. You can switch them via the dropdown.
- Never commit your real API keys. Use `.env` and environment variables.
