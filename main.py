
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import (
    Image, ImageTk, ImageOps, ImageEnhance,
    ImageDraw, ImageFont
)
import threading
import queue
import time
import math

# ============================================================
# PIXEL64 AI STUDIO 4.0
# ============================================================

APP_NAME = "PIXEL64 AI STUDIO 4.0"
MODEL_NAME = "yolov8s-worldv2.pt"

PREVIEW_SIZE = 480
TILE_SIZE = 768
TILE_OVERLAP = 0.25
MAX_TILES = 36
MAX_DETECTIONS = 200

BG = "#151820"
PANEL = "#202631"
PANEL_DARK = "#11151c"
ACCENT = "#65a8ff"
TEXT = "#f0f2f5"
MUTED = "#9da8b8"
GREEN = "#36d989"
ORANGE = "#ffb454"
RED = "#ff6b6b"

current_image = None
pixel_image = None
ai_annotated_image = None

original_preview = None
result_preview = None

model = None
detected_objects = []

ai_queue = queue.Queue()
ai_running = False
image_version = 0

# ============================================================
# OBJECT VOCABULARY
# ============================================================
# One clear name per category. Avoid overlapping labels such as
# child/adult/man/woman or shirt/t-shirt/jacket synonyms.
# The model does not identify people or infer personal attributes.

OBJECT_CLASSES = [
    # People
    "person",

    # Animals
    "dog", "cat", "horse", "cow", "sheep", "goat",
    "pig", "chicken", "duck", "bird", "rabbit",
    "hamster", "mouse", "rat", "snake", "lizard",
    "turtle", "frog", "fish", "butterfly", "bee",
    "spider", "deer", "fox", "wolf", "bear",
    "lion", "tiger", "elephant", "monkey", "zebra",

    # Vehicles
    "car", "truck", "bus", "van", "motorcycle",
    "bicycle", "scooter", "train", "airplane",
    "helicopter", "boat", "ship", "tractor",

    # Electronics
    "smartphone", "laptop", "desktop computer",
    "tablet", "keyboard", "computer mouse",
    "monitor", "television", "remote control",
    "camera", "headphones", "speaker", "microphone",
    "printer", "router", "game controller",
    "game console", "watch", "light bulb",

    # Furniture
    "chair", "table", "desk", "sofa", "bed",
    "wardrobe", "cabinet", "shelf", "bookcase",
    "lamp", "mirror", "refrigerator", "oven",
    "microwave", "washing machine",

    # Household objects
    "bottle", "cup", "glass", "plate", "bowl",
    "fork", "knife", "spoon", "pan", "pot",
    "kettle", "clock", "fan", "vacuum cleaner",
    "scissors", "umbrella", "backpack", "handbag",
    "suitcase", "book", "notebook", "pen",
    "pencil", "box", "basket", "trash can",

    # Tools
    "hammer", "screwdriver", "wrench", "pliers",
    "power drill", "saw", "toolbox", "ladder",
    "shovel", "axe", "flashlight",

    # Food
    "apple", "banana", "orange", "pear",
    "strawberry", "watermelon", "lemon", "grapes",
    "pineapple", "carrot", "potato", "tomato",
    "cucumber", "broccoli", "bread", "sandwich",
    "pizza", "hamburger", "hot dog", "cake",
    "donut", "cookie", "chocolate", "egg", "cheese",

    # Sports
    "football", "basketball", "tennis ball",
    "baseball", "volleyball", "soccer ball",
    "skateboard", "skis", "surfboard",
    "helmet", "tennis racket", "dumbbell",

    # Other objects and surroundings
    "tree", "flower", "potted plant", "grass",
    "building", "house", "door", "window",
    "stairs", "fence", "traffic light",
    "street sign", "fire hydrant",
    "fire extinguisher"
]


# ============================================================
# DRAG & DROP
# ============================================================

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")

    brightness = brightness_var.get()
    contrast = contrast_var.get()
    saturation = saturation_var.get()

    image = ImageEnhance.Brightness(image).enhance(
        max(0, 1 + brightness / 100)
    )
    image = ImageEnhance.Contrast(image).enhance(
        max(0, 1 + contrast / 100)
    )
    image = ImageEnhance.Color(image).enhance(
        max(0, 1 + saturation / 100)
    )

    width, height = image.size
    side = min(width, height)

    left = (width - side) // 2
    top = (height - side) // 2

    image = image.crop((
        left, top, left + side, top + side
    ))

    resolution = max(16, min(512, int(round(resolution_var.get()))))
    colors = max(2, min(64, int(round(colors_var.get()))))

    image = image.resize(
        (resolution, resolution),
        Image.Resampling.LANCZOS
    )

    dither = (
        Image.Dither.FLOYDSTEINBERG
        if dither_var.get()
        else Image.Dither.NONE
    )

    image = image.quantize(
        colors=colors,
        method=Image.Quantize.MEDIANCUT,
        dither=dither
    ).convert("RGB")

    return image


# ============================================================
# PREVIEWS
# ============================================================

def make_preview(image, size, pixelated=False):
    image = image.copy()

    scale = min(size / image.width, size / image.height)

    width = max(1, int(image.width * scale))
    height = max(1, int(image.height * scale))

    interpolation = (
        Image.Resampling.NEAREST
        if pixelated
        else Image.Resampling.LANCZOS
    )

    image = image.resize((width, height), interpolation)
    return ImageTk.PhotoImage(image)


def update_previews():
    global original_preview
    global result_preview
    global pixel_image

    if current_image is None:
        return

    try:
        display_image = (
            ai_annotated_image
            if ai_annotated_image is not None
            else current_image
        )

        original_preview = make_preview(
            display_image, PREVIEW_SIZE
        )

        original_label.configure(
            image=original_preview,
            text=""
        )

        pixel_image = process_image(current_image)

        result_preview = make_preview(
            pixel_image,
            PREVIEW_SIZE,
            pixelated=True
        )

        result_label.configure(
            image=result_preview,
            text=""
        )

        resolution = int(round(resolution_var.get()))
        colors = int(round(colors_var.get()))

        info_label.configure(
            text=f"{resolution} × {resolution} px  |  {colors} colors"
        )

    except Exception as error:
        status_label.configure(text=f"Processing error: {error}")


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(filename):
    global current_image
    global ai_annotated_image
    global detected_objects
    global image_version

    try:
        path = Path(filename)

        if not path.is_file():
            raise FileNotFoundError("Image file not found.")

        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.load()
            current_image = image.convert("RGB").copy()

        image_version += 1
        ai_annotated_image = None
        detected_objects = []

        filename_label.configure(text=path.name)

        update_previews()
        update_ai_results()

        status_label.configure(
            text=f"Loaded: {path.name}  |  {current_image.width} × {current_image.height}"
        )

    except Exception as error:
        messagebox.showerror(
            "Image error", str(error), parent=root
        )


def open_image():
    filename = filedialog.askopenfilename(
        parent=root,
        title="Select an image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff"),
            ("All files", "*.*")
        ]
    )

    if filename:
        load_image(filename)


# ============================================================
# AI MODEL
# ============================================================

def load_ai_model():
    global model

    if model is None:
        from ultralytics import YOLOWorld

        model = YOLOWorld(MODEL_NAME)
        model.set_classes(OBJECT_CLASSES)

    return model


# ============================================================
# TILING
# ============================================================

def generate_tiles(image, tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    width, height = image.size

    if width <= tile_size and height <= tile_size:
        return []

    step = max(1, int(tile_size * (1 - overlap)))

    def positions(length):
        if length <= tile_size:
            return [0]

        values = list(range(0, length - tile_size + 1, step))
        last = length - tile_size

        if not values or values[-1] != last:
            values.append(last)

        return values

    tiles = []

    for y in positions(height):
        for x in positions(width):
            right = min(x + tile_size, width)
            bottom = min(y + tile_size, height)

            crop = image.crop((x, y, right, bottom))
            tiles.append((crop, x, y))

    return tiles[:MAX_TILES]


# ============================================================
# DETECTION GEOMETRY
# ============================================================

def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def remove_duplicate_detections(objects, threshold=0.50):
    """
    Class-aware non-maximum suppression.
    Keeps the strongest detection when two boxes of the same
    category overlap heavily.
    """

    objects = sorted(
        objects,
        key=lambda item: item["confidence"],
        reverse=True
    )

    selected = []

    for obj in objects:
        duplicate = False

        for existing in selected:
            if obj["name"] != existing["name"]:
                continue

            if calculate_iou(
                obj["box"], existing["box"]
            ) >= threshold:
                duplicate = True
                break

        if not duplicate:
            selected.append(obj)

    # A second, conservative pass for nearly identical boxes
    # with different labels. It only removes extremely high overlap.
    final_objects = []

    for obj in selected:
        duplicate = False

        for existing in final_objects:
            if calculate_iou(
                obj["box"], existing["box"]
            ) >= 0.90:
                duplicate = True
                break

        if not duplicate:
            final_objects.append(obj)

    return final_objects[:MAX_DETECTIONS]


# ============================================================
# AI INFERENCE
# ============================================================

def extract_detections(results, ai_model, offset_x=0, offset_y=0):
    detections = []

    if not results:
        return detections

    result = results[0]

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        score = float(box.conf[0])

        name = ai_model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "name": str(name),
            "confidence": score,
            "box": (
                x1 + offset_x,
                y1 + offset_y,
                x2 + offset_x,
                y2 + offset_y
            )
        })

    return detections


def analyze_image_worker(image, minimum_confidence, small_objects):
    ai_model = load_ai_model()

    detections = []

    # The model can consider lower scores internally.
    # The user-selected confidence is applied afterwards.
    inference_confidence = max(
        0.08,
        min(0.45, minimum_confidence * 0.65)
    )

    # Full image pass
    results = ai_model.predict(
        source=image,
        imgsz=1280,
        conf=inference_confidence,
        iou=0.45,
        max_det=300,
        verbose=False
    )

    detections.extend(
        extract_detections(results, ai_model)
    )

    # Extra pass for small objects
    tiles = generate_tiles(image) if small_objects else []

    for index, (crop, offset_x, offset_y) in enumerate(tiles, 1):
        ai_queue.put(("progress", f"Analyzing detail {index}/{len(tiles)}..."))

        tile_results = ai_model.predict(
            source=crop,
            imgsz=960,
            conf=inference_confidence,
            iou=0.45,
            max_det=200,
            verbose=False
        )

        detections.extend(
            extract_detections(
                tile_results,
                ai_model,
                offset_x,
                offset_y
            )
        )

    # Apply user-selected confidence threshold
    detections = [
        item for item in detections
        if item["confidence"] >= minimum_confidence
    ]

    detections = remove_duplicate_detections(
        detections,
        threshold=0.50
    )

    return detections


# ============================================================
# DRAWING ANNOTATIONS
# ============================================================

def confidence_color(score):
    if score >= 0.70:
        return "#21b96b"

    if score >= 0.40:
        return "#e69a32"

    return "#df5757"


def draw_ai_results(image, objects):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    font_size = max(13, min(28, image.width // 45))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    line_width = max(2, image.width // 350)

    for obj in objects:
        x1, y1, x2, y2 = obj["box"]

        x1 = max(0, min(image.width - 1, int(x1)))
        y1 = max(0, min(image.height - 1, int(y1)))
        x2 = max(0, min(image.width - 1, int(x2)))
        y2 = max(0, min(image.height - 1, int(y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        color = confidence_color(obj["confidence"])

        label = (
            f"{obj['name']} "
            f"{obj['confidence'] * 100:.0f}%"
        )

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=line_width
        )

        bbox = draw.textbbox((x1, y1), label, font=font)
        label_height = bbox[3] - bbox[1]

        label_y = max(0, y1 - label_height - 7)

        label_box = draw.textbbox(
            (x1, label_y), label, font=font
        )

        draw.rectangle(label_box, fill=color)
        draw.text(
            (x1, label_y),
            label,
            fill="white",
            font=font
        )

    return annotated


# ============================================================
# RUN AI
# ============================================================

def run_ai():
    global ai_running

    if current_image is None:
        messagebox.showwarning(
            "No image",
            "Load an image before using AI.",
            parent=root
        )
        return

    if ai_running:
        return

    ai_running = True
    analyze_button.configure(state="disabled")

    image_copy = current_image.copy()
    version_at_start = image_version

    confidence = confidence_var.get() / 100
    small_objects = small_objects_var.get()

    status_label.configure(
        text="AI is loading and analyzing the image..."
    )

    progress_bar.start(12)

    start_time = time.perf_counter()

    def worker():
        try:
            objects = analyze_image_worker(
                image_copy,
                confidence,
                small_objects
            )

            annotated = draw_ai_results(image_copy, objects)
            elapsed = time.perf_counter() - start_time

            ai_queue.put((
                "success",
                objects,
                annotated,
                elapsed,
                version_at_start
            ))

        except Exception as error:
            ai_queue.put((
                "error",
                str(error),
                version_at_start
            ))

    threading.Thread(
        target=worker,
        daemon=True
    ).start()

    root.after(150, check_ai_queue)


def check_ai_queue():
    global ai_running
    global detected_objects
    global ai_annotated_image

    try:
        result = ai_queue.get_nowait()

    except queue.Empty:
        if ai_running:
            root.after(150, check_ai_queue)
        return

    if result[0] == "progress":
        status_label.configure(text=result[1])
        root.after(100, check_ai_queue)
        return

    ai_running = False
    progress_bar.stop()
    analyze_button.configure(state="normal")

    if result[0] == "success":
        _, objects, annotated, elapsed, version = result

        # Ignore results belonging to an image that was replaced.
        if version != image_version:
            status_label.configure(
                text="Old analysis discarded because a new image was opened."
            )
            return

        detected_objects = objects
        ai_annotated_image = annotated

        update_previews()
        update_ai_results()

        status_label.configure(
            text=(
                f"Analysis complete: {len(objects)} detections "
                f"in {elapsed:.1f} seconds."
            )
        )

    else:
        _, error, version = result

        if version == image_version:
            status_label.configure(text="AI recognition failed.")

            messagebox.showerror(
                "AI error",
                error,
                parent=root
            )


# ============================================================
# FILTERED RESULTS AND STATISTICS
# ============================================================

def get_filtered_objects():
    if not detected_objects:
        return []

    selected_category = category_filter_var.get()
    minimum = result_filter_var.get() / 100

    filtered = [
        obj for obj in detected_objects
        if obj["confidence"] >= minimum
    ]

    if selected_category != "All categories":
        filtered = [
            obj for obj in filtered
            if obj["name"] == selected_category
        ]

    return filtered


def update_category_filter():
    categories = sorted({
        obj["name"] for obj in detected_objects
    })

    options = ["All categories"] + categories

    category_combo.configure(values=options)
    category_filter_var.set("All categories")


def update_ai_results():
    ai_results.configure(state="normal")
    ai_results.delete("1.0", "end")

    if current_image is None:
        ai_results.insert("end", "Load an image to begin.")
        stats_label.configure(text="No analysis yet")
        ai_results.configure(state="disabled")
        return

    if not detected_objects:
        ai_results.insert(
            "end",
            "No results yet, or no objects passed the confidence threshold.\n"
            "Try lowering the AI confidence setting."
        )
        stats_label.configure(text="0 detections")
        update_category_filter()
        ai_results.configure(state="disabled")
        return

    update_category_filter()

    filtered = get_filtered_objects()

    counts = {}
    for obj in detected_objects:
        counts[obj["name"]] = counts.get(obj["name"], 0) + 1

    summary = ", ".join(
        f"{name}: {count}"
        for name, count in sorted(counts.items())
    )

    ai_results.insert(
        "end",
        f"TOTAL DETECTIONS: {len(detected_objects)}\n"
        f"CATEGORIES: {len(counts)}\n"
        f"FILTERED RESULTS: {len(filtered)}\n\n"
        f"CATEGORY SUMMARY:\n{summary}\n\n"
        f"DETECTION LIST:\n"
    )

    for index, obj in enumerate(filtered, 1):
        score = obj["confidence"] * 100

        if score >= 70:
            quality = "HIGH"
        elif score >= 40:
            quality = "MEDIUM"
        else:
            quality = "LOW"

        ai_results.insert(
            "end",
            f"{index:03d}. {obj['name']:<22} "
            f"{score:5.1f}%  [{quality}]\n"
        )

    stats_label.configure(
        text=(
            f"{len(detected_objects)} detections  |  "
            f"{len(counts)} categories  |  "
            f"{len(filtered)} visible"
        )
    )

    ai_results.configure(state="disabled")


def refresh_result_filter(value=None):
    if detected_objects:
        update_ai_results()


def clear_ai():
    global ai_annotated_image
    global detected_objects

    if ai_running:
        messagebox.showinfo(
            "AI running",
            "Wait until the current analysis is finished.",
            parent=root
        )
        return

    ai_annotated_image = None
    detected_objects = []

    update_previews()
    update_ai_results()

    status_label.configure(text="AI results cleared.")


# ============================================================
# SAVE PIXEL ART
# ============================================================

def save_image():
    if pixel_image is None:
        messagebox.showwarning(
            "No image",
            "Load an image first.",
            parent=root
        )
        return

    filename = filedialog.asksaveasfilename(
        parent=root,
        title="Save pixel art",
        defaultextension=".png",
        initialfile="pixel64.png",
        filetypes=[("PNG image", "*.png")]
    )

    if not filename:
        return

    try:
        pixel_image.save(filename, format="PNG")
        status_label.configure(
            text=f"Pixel art saved: {Path(filename).name}"
        )

    except Exception as error:
        messagebox.showerror(
            "Save error", str(error), parent=root
        )


# ============================================================
# SAVE AI IMAGE
# ============================================================

def save_ai_image():
    if ai_annotated_image is None:
        messagebox.showwarning(
            "No AI result",
            "Analyze an image first.",
            parent=root
        )
        return

    filename = filedialog.asksaveasfilename(
        parent=root,
        title="Save AI detection",
        defaultextension=".png",
        initialfile="ai_detection.png",
        filetypes=[("PNG image", "*.png")]
    )

    if not filename:
        return

    try:
        ai_annotated_image.save(filename, format="PNG")
        status_label.configure(
            text=f"AI image saved: {Path(filename).name}"
        )

    except Exception as error:
        messagebox.showerror(
            "Save error", str(error), parent=root
        )


# ============================================================
# SAVE DETECTION REPORT
# ============================================================

def save_report():
    if not detected_objects:
        messagebox.showwarning(
            "No results",
            "Run AI analysis first.",
            parent=root
        )
        return

    filename = filedialog.asksaveasfilename(
        parent=root,
        title="Save detection report",
        defaultextension=".txt",
        initialfile="ai_report.txt",
        filetypes=[("Text file", "*.txt")]
    )

    if not filename:
        return

    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write("PIXEL64 AI STUDIO 4.0 - DETECTION REPORT\n")
            file.write("=" * 55 + "\n\n")

            file.write(f"Image: {filename_label.cget('text')}\n")
            file.write(f"Model: {MODEL_NAME}\n")
            file.write(f"Detections: {len(detected_objects)}\n\n")

            for index, obj in enumerate(detected_objects, 1):
                file.write(
                    f"{index}. {obj['name']} - "
                    f"{obj['confidence'] * 100:.1f}%\n"
                )

                file.write(
                    f"   Box: {tuple(round(v, 1) for v in obj['box'])}\n"
                )

        status_label.configure(
            text=f"Report saved: {Path(filename).name}"
        )

    except Exception as error:
        messagebox.showerror(
            "Report error", str(error), parent=root
        )


# ============================================================
# SETTINGS
# ============================================================

def settings_changed(value=None):
    if current_image is not None:
        update_previews()


# ============================================================
# DRAG & DROP
# ============================================================

def drop_image(event):
    try:
        files = root.tk.splitlist(event.data)

        if files:
            load_image(files[0])

    except Exception as error:
        messagebox.showerror(
            "Drop error", str(error), parent=root
        )


# ============================================================
# WINDOW
# ============================================================

if DND_AVAILABLE:
    root = TkinterDnD.Tk()
else:
    root = tk.Tk()

root.title(APP_NAME)
root.geometry("1450x1120")
root.minsize(1120, 900)
root.configure(bg=BG)


# ============================================================
# STYLES
# ============================================================

style = ttk.Style()
style.theme_use("clam")

style.configure(
    "TButton",
    font=("Segoe UI", 10, "bold"),
    padding=9
)

style.configure(
    "TScale",
    background=PANEL,
    troughcolor="#394252"
)

style.configure(
    "TCheckbutton",
    background=PANEL,
    foreground=TEXT,
    font=("Segoe UI", 9)
)

style.configure(
    "TCombobox",
    fieldbackground=PANEL_DARK,
    background=PANEL,
    foreground=TEXT
)

style.configure(
    "Horizontal.TProgressbar",
    troughcolor="#303846",
    background=ACCENT
)


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(root, bg=BG)
header.pack(fill="x", padx=25, pady=(15, 8))

tk.Label(
    header,
    text="PIXEL64",
    font=("Segoe UI", 30, "bold"),
    fg=ACCENT,
    bg=BG
).pack(side="left")

tk.Label(
    header,
    text="  AI PIXEL ART STUDIO 4.0",
    font=("Segoe UI", 12, "bold"),
    fg=MUTED,
    bg=BG
).pack(side="left", pady=(14, 0))


# ============================================================
# PREVIEW AREA
# ============================================================

main = tk.Frame(root, bg=BG)
main.pack(fill="both", expand=True, padx=25, pady=8)

main.columnconfigure(0, weight=1)
main.columnconfigure(1, weight=1)
main.rowconfigure(0, weight=1)

original_panel = tk.Frame(
    main, bg=PANEL, padx=12, pady=12
)
original_panel.grid(
    row=0, column=0,
    sticky="nsew", padx=(0, 8)
)

tk.Label(
    original_panel,
    text="ORIGINAL / AI DETECTION",
    font=("Segoe UI", 12, "bold"),
    fg=TEXT,
    bg=PANEL
).pack(pady=(0, 10))

original_label = tk.Label(
    original_panel,
    text="OPEN IMAGE\nor drag a photo here",
    fg=MUTED,
    bg=PANEL_DARK,
    font=("Segoe UI", 11)
)

original_label.pack(expand=True, fill="both")

result_panel = tk.Frame(
    main, bg=PANEL, padx=12, pady=12
)
result_panel.grid(
    row=0, column=1,
    sticky="nsew", padx=(8, 0)
)

tk.Label(
    result_panel,
    text="PIXEL ART RESULT",
    font=("Segoe UI", 12, "bold"),
    fg=ACCENT,
    bg=PANEL
).pack(pady=(0, 10))

result_label = tk.Label(
    result_panel,
    text="Pixel art will appear here",
    fg=MUTED,
    bg=PANEL_DARK,
    font=("Segoe UI", 11)
)

result_label.pack(expand=True, fill="both")


# ============================================================
# CONFIGURATION VARIABLES
# ============================================================

resolution_var = tk.DoubleVar(value=64)
colors_var = tk.DoubleVar(value=16)

brightness_var = tk.DoubleVar(value=0)
contrast_var = tk.DoubleVar(value=0)
saturation_var = tk.DoubleVar(value=0)

confidence_var = tk.DoubleVar(value=35)
result_filter_var = tk.DoubleVar(value=35)

dither_var = tk.BooleanVar(value=False)
small_objects_var = tk.BooleanVar(value=True)

category_filter_var = tk.StringVar(value="All categories")


# ============================================================
# CONFIGURATION PANEL
# ============================================================

controls = tk.Frame(
    root, bg=PANEL, padx=18, pady=12
)

controls.pack(fill="x", padx=25, pady=8)


def add_slider(parent, title, variable, minimum, maximum,
               callback=None, suffix=""):

    row = tk.Frame(parent, bg=PANEL)
    row.pack(fill="x", pady=4)

    tk.Label(
        row,
        text=title,
        fg=TEXT,
        bg=PANEL,
        font=("Segoe UI", 9, "bold"),
        width=20,
        anchor="w"
    ).pack(side="left")

    value_label = tk.Label(
        row,
        text=f"{int(variable.get())}{suffix}",
        fg=ACCENT,
        bg=PANEL,
        width=7
    )
    value_label.pack(side="right")

    def changed(value):
        number = round(float(value))
        value_label.configure(text=f"{number}{suffix}")

        if callback:
            callback(value)
        else:
            settings_changed()

    slider = ttk.Scale(
        row,
        from_=minimum,
        to=maximum,
        variable=variable,
        command=changed
    )

    slider.pack(
        side="left",
        fill="x",
        expand=True,
        padx=10
    )

    return slider


settings_grid = tk.Frame(controls, bg=PANEL)
settings_grid.pack(fill="x")

settings_grid.columnconfigure(0, weight=1)
settings_grid.columnconfigure(1, weight=1)

left_settings = tk.Frame(settings_grid, bg=PANEL)
left_settings.grid(
    row=0, column=0,
    sticky="ew", padx=(0, 15)
)

right_settings = tk.Frame(settings_grid, bg=PANEL)
right_settings.grid(
    row=0, column=1,
    sticky="ew", padx=(15, 0)
)

add_slider(
    left_settings, "Resolution",
    resolution_var, 16, 512
)

add_slider(
    left_settings, "Color palette",
    colors_var, 2, 64
)

add_slider(
    left_settings, "Brightness",
    brightness_var, -100, 100
)

add_slider(
    right_settings, "Contrast",
    contrast_var, -100, 100
)

add_slider(
    right_settings, "Saturation",
    saturation_var, -100, 100
)

add_slider(
    right_settings, "AI confidence",
    confidence_var, 10, 90,
    suffix="%"
)

ttk.Checkbutton(
    controls,
    text="Enable dithering (retro color effect)",
    variable=dither_var,
    command=settings_changed
).pack(anchor="w", pady=(8, 0))

ttk.Checkbutton(
    controls,
    text="Enhanced small-object detection (slower)",
    variable=small_objects_var
).pack(anchor="w", pady=(5, 0))


# ============================================================
# BUTTONS
# ============================================================

buttons = tk.Frame(root, bg=BG)
buttons.pack(fill="x", padx=25, pady=8)

ttk.Button(
    buttons,
    text="OPEN IMAGE",
    command=open_image
).pack(side="left", padx=(0, 7))

analyze_button = ttk.Button(
    buttons,
    text="ANALYZE IMAGE (AI)",
    command=run_ai
)
analyze_button.pack(side="left", padx=7)

ttk.Button(
    buttons,
    text="CLEAR AI",
    command=clear_ai
).pack(side="left", padx=7)

ttk.Button(
    buttons,
    text="SAVE PIXEL PNG",
    command=save_image
).pack(side="left", padx=7)

ttk.Button(
    buttons,
    text="SAVE AI IMAGE",
    command=save_ai_image
).pack(side="left", padx=7)

ttk.Button(
    buttons,
    text="SAVE REPORT",
    command=save_report
).pack(side="left", padx=7)


# ============================================================
# PROGRESS
# ============================================================

progress_bar = ttk.Progressbar(
    root,
    mode="indeterminate",
    style="Horizontal.TProgressbar"
)

progress_bar.pack(
    fill="x",
    padx=25,
    pady=(3, 5)
)


# ============================================================
# AI RESULTS PANEL
# ============================================================

ai_panel = tk.Frame(
    root, bg=PANEL, padx=15, pady=10
)

ai_panel.pack(
    fill="both",
    expand=False,
    padx=25,
    pady=5
)

results_header = tk.Frame(ai_panel, bg=PANEL)
results_header.pack(fill="x")

tk.Label(
    results_header,
    text="AI DETECTION RESULTS",
    font=("Segoe UI", 11, "bold"),
    fg=ACCENT,
    bg=PANEL
).pack(side="left")

stats_label = tk.Label(
    results_header,
    text="No analysis yet",
    font=("Segoe UI", 9),
    fg=MUTED,
    bg=PANEL
)
stats_label.pack(side="right")

filter_row = tk.Frame(ai_panel, bg=PANEL)
filter_row.pack(fill="x", pady=(8, 5))

tk.Label(
    filter_row,
    text="Category:",
    fg=TEXT,
    bg=PANEL
).pack(side="left")

category_combo = ttk.Combobox(
    filter_row,
    textvariable=category_filter_var,
    values=["All categories"],
    state="readonly",
    width=24
)

category_combo.pack(side="left", padx=(7, 15))
category_combo.bind(
    "<<ComboboxSelected>>",
    lambda event: update_ai_results()
)

tk.Label(
    filter_row,
    text="Minimum result score:",
    fg=TEXT,
    bg=PANEL
).pack(side="left")

add_slider(
    ai_panel,
    "Result filter",
    result_filter_var,
    0,
    90,
    callback=refresh_result_filter,
    suffix="%"
)

results_frame = tk.Frame(ai_panel, bg=PANEL)
results_frame.pack(fill="both", expand=True, pady=(5, 0))

ai_results = tk.Text(
    results_frame,
    height=8,
    bg=PANEL_DARK,
    fg=TEXT,
    insertbackground=TEXT,
    font=("Consolas", 9),
    relief="flat",
    wrap="none"
)

ai_scrollbar = ttk.Scrollbar(
    results_frame,
    orient="vertical",
    command=ai_results.yview
)

ai_results.configure(
    yscrollcommand=ai_scrollbar.set
)

ai_results.pack(
    side="left",
    fill="both",
    expand=True
)

ai_scrollbar.pack(side="right", fill="y")

ai_results.insert(
    "end",
    "Load an image and click ANALYZE IMAGE (AI)."
)

ai_results.configure(state="disabled")


# ============================================================
# FOOTER
# ============================================================

footer = tk.Frame(root, bg=BG)
footer.pack(fill="x", padx=25, pady=8)

filename_label = tk.Label(
    footer,
    text="No image selected",
    fg=MUTED,
    bg=BG
)
filename_label.pack(side="left")

info_label = tk.Label(
    footer,
    text="64 × 64 px | 16 colors",
    fg=ACCENT,
    bg=BG
)
info_label.pack(side="right")

status_label = tk.Label(
    root,
    text="Ready.",
    fg=MUTED,
    bg=BG,
    anchor="w"
)

status_label.pack(
    fill="x",
    padx=25,
    pady=(0, 10)
)


# ============================================================
# DRAG & DROP REGISTRATION
# ============================================================

if DND_AVAILABLE:
    for widget in (
        root,
        original_panel,
        original_label,
        result_panel,
        result_label
    ):
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind("<<Drop>>", drop_image)
else:
    status_label.configure(
        text="Drag & drop unavailable. Use OPEN IMAGE."
    )


# ============================================================
# START
# ============================================================

root.mainloop()
