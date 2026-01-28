import argparse
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
import torch

import sys
import importlib
import importlib.util

sys.path.append(".")

from model.models import UNet
from scripts.test_functions import process_image

_dnd_spec = importlib.util.find_spec("tkinterdnd2")
if _dnd_spec:
    tkinterdnd2 = importlib.import_module("tkinterdnd2")
    DND_FILES = tkinterdnd2.DND_FILES
    TkinterDnD = tkinterdnd2.TkinterDnD
    HAS_DND = True
else:
    HAS_DND = False
    DND_FILES = None
    TkinterDnD = None


class FaceReagingApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Face Re-Aging (Windows)")
        self.root.geometry("720x520")
        self.root.minsize(640, 480)

        self.model_path = tk.StringVar(value=model_path)
        self.input_path = tk.StringVar(value="")
        self.output_path = tk.StringVar(value="")
        self.source_age = tk.IntVar(value=30)
        self.target_age = tk.IntVar(value=60)
        self.status_text = tk.StringVar(value="Ready")

        self._model = None
        self._model_device = None
        self._model_path_loaded = None
        self._preview_image = None

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        model_frame = ttk.LabelFrame(main, text="Model")
        model_frame.pack(fill=tk.X, pady=6)

        ttk.Label(model_frame, text="Model path:").grid(row=0, column=0, sticky=tk.W, padx=6, pady=6)
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path, width=60)
        model_entry.grid(row=0, column=1, sticky=tk.W + tk.E, padx=6, pady=6)
        ttk.Button(model_frame, text="Browse", command=self._browse_model).grid(
            row=0, column=2, sticky=tk.W, padx=6, pady=6
        )
        model_frame.columnconfigure(1, weight=1)

        input_frame = ttk.LabelFrame(main, text="Input image")
        input_frame.pack(fill=tk.X, pady=6)

        ttk.Label(input_frame, text="Image path:").grid(row=0, column=0, sticky=tk.W, padx=6, pady=6)
        input_entry = ttk.Entry(input_frame, textvariable=self.input_path, width=60)
        input_entry.grid(row=0, column=1, sticky=tk.W + tk.E, padx=6, pady=6)
        ttk.Button(input_frame, text="Browse", command=self._browse_input).grid(
            row=0, column=2, sticky=tk.W, padx=6, pady=6
        )
        input_frame.columnconfigure(1, weight=1)

        if HAS_DND:
            drop_label = ttk.Label(input_frame, text="Drag & drop an image onto this window")
            drop_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=6, pady=(0, 6))
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", self._handle_drop)
        else:
            hint_label = ttk.Label(
                input_frame,
                text="Tip: install tkinterdnd2 for drag & drop support",
                foreground="#555",
            )
            hint_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=6, pady=(0, 6))

        params_frame = ttk.LabelFrame(main, text="Aging parameters")
        params_frame.pack(fill=tk.X, pady=6)

        ttk.Label(params_frame, text="Source age:").grid(row=0, column=0, sticky=tk.W, padx=6, pady=6)
        ttk.Spinbox(params_frame, from_=1, to=100, textvariable=self.source_age, width=8).grid(
            row=0, column=1, sticky=tk.W, padx=6, pady=6
        )
        ttk.Label(params_frame, text="Target age:").grid(row=0, column=2, sticky=tk.W, padx=6, pady=6)
        ttk.Spinbox(params_frame, from_=1, to=100, textvariable=self.target_age, width=8).grid(
            row=0, column=3, sticky=tk.W, padx=6, pady=6
        )

        output_frame = ttk.LabelFrame(main, text="Output")
        output_frame.pack(fill=tk.X, pady=6)

        ttk.Label(output_frame, text="Output path:").grid(row=0, column=0, sticky=tk.W, padx=6, pady=6)
        output_entry = ttk.Entry(output_frame, textvariable=self.output_path, width=60)
        output_entry.grid(row=0, column=1, sticky=tk.W + tk.E, padx=6, pady=6)
        ttk.Button(output_frame, text="Save as", command=self._browse_output).grid(
            row=0, column=2, sticky=tk.W, padx=6, pady=6
        )
        output_frame.columnconfigure(1, weight=1)

        preview_frame = ttk.LabelFrame(main, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=6)
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        footer = ttk.Frame(main)
        footer.pack(fill=tk.X, pady=(6, 0))
        self.generate_button = ttk.Button(footer, text="Generate", command=self._start_generation)
        self.generate_button.pack(side=tk.RIGHT, padx=6)
        status_label = ttk.Label(footer, textvariable=self.status_text)
        status_label.pack(side=tk.LEFT, padx=6)

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")],
        )
        if path:
            self.model_path.set(path)

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select input image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")],
        )
        if path:
            self.input_path.set(path)
            self._load_preview(path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save output image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("All files", "*.*")],
        )
        if path:
            self.output_path.set(path)

    def _handle_drop(self, event):
        raw_path = event.data.strip()
        if raw_path.startswith("{") and raw_path.endswith("}"):
            raw_path = raw_path[1:-1]
        self.input_path.set(raw_path)
        self._load_preview(raw_path)

    def _load_preview(self, path):
        if not os.path.exists(path):
            return
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((480, 320))
            self._preview_image = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self._preview_image)
        except Exception:
            self.preview_label.configure(image="")
            self._preview_image = None

    def _start_generation(self):
        if self.generate_button["state"] == tk.DISABLED:
            return
        thread = threading.Thread(target=self._generate, daemon=True)
        thread.start()

    def _validate_inputs(self):
        if not os.path.exists(self.model_path.get()):
            messagebox.showerror("Missing model", "Please select a valid model file (.pth).")
            return False
        if not os.path.exists(self.input_path.get()):
            messagebox.showerror("Missing input", "Please select a valid input image.")
            return False
        if not (1 <= self.source_age.get() <= 100):
            messagebox.showerror("Invalid age", "Source age must be between 1 and 100.")
            return False
        if not (1 <= self.target_age.get() <= 100):
            messagebox.showerror("Invalid age", "Target age must be between 1 and 100.")
            return False
        return True

    def _ensure_model(self):
        if self._model is not None and self._model_path_loaded == self.model_path.get():
            return
        self.status_text.set("Loading model...")
        self.root.update_idletasks()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = UNet().to(device)
        model.load_state_dict(torch.load(self.model_path.get(), map_location=device))
        model.eval()
        self._model = model
        self._model_device = device
        self._model_path_loaded = self.model_path.get()

    def _generate(self):
        if not self._validate_inputs():
            return
        self.generate_button.configure(state=tk.DISABLED)
        try:
            self._ensure_model()
            self.status_text.set("Running inference...")
            self.root.update_idletasks()

            image = Image.open(self.input_path.get()).convert("RGB")
            result = process_image(
                self._model,
                image,
                video=False,
                source_age=self.source_age.get(),
                target_age=self.target_age.get(),
            )
            output_path = self.output_path.get().strip()
            if not output_path:
                base, _ = os.path.splitext(self.input_path.get())
                output_path = f"{base}_aged_{self.target_age.get()}.png"
                self.output_path.set(output_path)

            result.save(output_path)
            self._load_preview(output_path)
            self.status_text.set("Done.")
            messagebox.showinfo("Done", f"Saved output to:\n{output_path}")
        except Exception as exc:
            self.status_text.set("Error")
            messagebox.showerror("Error", str(exc))
        finally:
            self.generate_button.configure(state=tk.NORMAL)


def main():
    parser = argparse.ArgumentParser(description="Windows GUI wrapper for face re-aging")
    parser.add_argument(
        "--model_path",
        type=str,
        default="best_unet_model.pth",
        help="Path to best_unet_model.pth",
    )
    args = parser.parse_args()

    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    app = FaceReagingApp(root, args.model_path)
    root.mainloop()


if __name__ == "__main__":
    main()
