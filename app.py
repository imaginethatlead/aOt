import datetime
import json
import os
import pathlib
import shlex
import subprocess
import gradio as gr

DATA_ROOT = pathlib.Path(os.environ.get("DATA_ROOT", "/data"))
INCOMING_DIR = DATA_ROOT / "incoming_videos"
STANDARDIZED_DIR = DATA_ROOT / "standardized_videos"
DATASET_DIR = DATA_ROOT / "dataset" / "runs"
DEFAULT_AVOCADO_CMD = os.environ.get("AVOCADO_CMD", "")


def ensure_dirs() -> None:
    for directory in (INCOMING_DIR, STANDARDIZED_DIR, DATASET_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_uploads(files: list[gr.FileData]) -> list[pathlib.Path]:
    saved_paths: list[pathlib.Path] = []
    for file_data in files:
        src_path = pathlib.Path(file_data.path)
        dest_path = INCOMING_DIR / src_path.name
        dest_path.write_bytes(src_path.read_bytes())
        saved_paths.append(dest_path)
    return saved_paths


def standardize_video(input_path: pathlib.Path) -> pathlib.Path:
    output_path = STANDARDIZED_DIR / f"{input_path.stem}_720p.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        "scale=-2:720,format=yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def run_avocado(
    input_path: pathlib.Path, output_path: pathlib.Path, avocado_cmd: str
) -> dict:
    if not avocado_cmd:
        return {
            "success": False,
            "error": "AVOCADO_CMD is not set. Provide a command in the UI or as an env var.",
        }

    formatted_cmd = avocado_cmd.format(
        input=shlex.quote(str(input_path)),
        output=shlex.quote(str(output_path)),
    )
    result = subprocess.run(
        formatted_cmd,
        shell=True,
        capture_output=True,
        text=True,
    )

    output_text = ""
    if output_path.exists():
        output_text = output_path.read_text(encoding="utf-8")

    return {
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output": output_text,
    }


def write_annotations(run_dir: pathlib.Path, records: list[dict]) -> pathlib.Path:
    output_path = run_dir / "annotations.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def write_manifest(run_dir: pathlib.Path, records: list[dict]) -> pathlib.Path:
    manifest_path = run_dir / "manifest.json"
    manifest_payload = {
        "run_label": run_dir.name,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "count": len(records),
        "records": records,
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def process_videos(files: list[gr.FileData], avocado_cmd: str) -> str:
    ensure_dirs()
    if not files:
        return "No files uploaded."

    run_label = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = DATASET_DIR / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = save_uploads(files)
    records: list[dict] = []

    for input_path in saved_paths:
        standardized_path = standardize_video(input_path)
        avocado_output_path = run_dir / f"{standardized_path.stem}_avocado.json"
        avocado_result = run_avocado(standardized_path, avocado_output_path, avocado_cmd)

        record = {
            "run_label": run_label,
            "source_file": str(input_path),
            "standardized_file": str(standardized_path),
            "avocado_output_file": str(avocado_output_path),
            "avocado": avocado_result,
            "processed_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        records.append(record)

    annotations_path = write_annotations(run_dir, records)
    manifest_path = write_manifest(run_dir, records)

    return (
        "Processing complete.\n"
        f"Run label: {run_label}\n"
        f"Annotations: {annotations_path}\n"
        f"Manifest: {manifest_path}"
    )


def build_app() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown(
            "# Avocado On Toast\n"
            "Upload TikTok videos, standardize with ffmpeg, run AVoCaDO, "
            "and save JSONL outputs into a run-labeled dataset folder."
        )
        with gr.Row():
            files = gr.File(label="Upload videos", file_count="multiple")
            avocado_cmd = gr.Textbox(
                label="AVoCaDO command",
                value=DEFAULT_AVOCADO_CMD,
                placeholder=(
                    "Example: python -m avocado_captioner.cli "
                    "--input {input} --output {output}"
                ),
            )
        run_button = gr.Button("Process videos")
        output = gr.Textbox(label="Status", lines=6)

        run_button.click(process_videos, inputs=[files, avocado_cmd], outputs=output)

        gr.Markdown(
            "**Output location:** `/data/dataset/runs/<run_label>/annotations.jsonl`\n\n"
            "Set `AVOCADO_CMD` in your Space settings or in the textbox above. "
            "The command must accept `{input}` and `{output}` placeholders."
        )
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
