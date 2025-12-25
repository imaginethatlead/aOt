---
title: Avocado On Toast
emoji: 🥑
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---

# Avocado On Toast

Gradio Space that standardizes uploaded videos with ffmpeg, runs AVoCaDO, and
writes JSONL outputs into a run-labeled dataset folder.

## Usage (Hugging Face Space)

1. Create a new Gradio Space and connect it to this repo.
2. Ensure `ffmpeg` and AVoCaDO are available in the Space image.
3. Set `AVOCADO_CMD` as a Space secret or enter it in the UI textbox.
   - The command must accept `{input}` and `{output}` placeholders.
   - Example:
     ```bash
     python -m avocado_captioner.cli --input {input} --output {output}
     ```
4. Launch the Space, upload videos, and click **Process videos**.

Outputs are written to:

```
/data/dataset/runs/<run_label>/annotations.jsonl
```

Each run produces `annotations.jsonl` and `manifest.json` under the run label.

## Configuration

- `DATA_ROOT` (default: `/data`) controls the root folder for uploads and output.
- `AVOCADO_CMD` defines the AVoCaDO command used during processing.

## Notes

The standardization step uses:
- 720p scaling (`scale=-2:720`)
- H.264 video (`libx264`, `crf=23`)
- AAC audio (`128k`)

Adjust these settings in `app.py` if you want a different quality/compute tradeoff.
