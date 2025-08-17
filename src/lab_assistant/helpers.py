import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import base64
from duckduckgo_search import DDGS
import csv
from dotenv import load_dotenv

load_dotenv()

def web_search(query: str, max_results: int = 5):
    """Simple DuckDuckGo text search."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
    return results

def get_current_datetime():
    """Return a human-readable current datetime string."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def image_to_base64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return f"Error encoding image: {e}"

def capture_image(year=None, month=None, day=None, hour=None, minute=None, camera_url: str=None):
    """
    Capture an image from a camera URL and save it with a timestamp-based filename.
    If a full timestamp is given and a file with that name exists, return its path.
    """
    def check_image_exists(filename):
        return os.path.isfile(filename)

    try:
        if all(param is not None for param in [year, month, day, hour, minute]):
            filename = f"{year}_{month}_{day}_{hour}_{minute}.jpg"
            filepath = os.path.join(os.getcwd(), filename)
            if check_image_exists(filepath):
                return filepath
        else:
            now = datetime.now()
            filename = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}.jpg"
            filepath = os.path.join(os.getcwd(), filename)

        # Allow override via parameter or env
        url = camera_url or os.getenv("CAMERA_URL", "http://10.31.160.186/cam-hi.jpg")

        resp = urllib.request.urlopen(url, timeout=10)
        imgnp = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, cv2.IMREAD_COLOR)
        if im is None:
            return "Error capturing image: decode returned None"
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        plt.imshow(im_rgb)
        plt.axis("off")
        plt.show()

        cv2.imwrite(filepath, im)
        return filepath
    except Exception as e:
        return f"Error capturing image: {e}"

def monitor_equipment_status(equipment_type: str = "liquid handler"):
    """
    Show the timestamped image in a given equipment folder closest to now.
    Folders should live in CWD under the equipment name and contain '*.jpg' with 'Y_M_D_H_M.jpg' names.
    """
    try:
        now = datetime.now()
        directory = os.path.join(os.getcwd(), equipment_type)
        files = os.listdir(directory)
        images = [f for f in files if f.endswith(".jpg") and len(f.split("_")) == 5]

        image_datetimes = []
        for image in images:
            try:
                year, month, day, hour, minute = map(int, image[:-4].split("_"))
                image_datetime = datetime(year, month, day, hour, minute)
                image_datetimes.append((image_datetime, image))
            except ValueError:
                continue

        if not image_datetimes:
            return f"No valid images found in {equipment_type} folder."

        closest_image = min(image_datetimes, key=lambda x: abs(now - x[0]))
        closest_image_path = os.path.join(directory, closest_image[1])

        img = cv2.imread(closest_image_path)
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(im_rgb)
        plt.axis("off")
        plt.show()
        return closest_image_path
    except Exception as e:
        return f"Error monitoring equipment status: {e}"

def generate_tecan_csv(
    plate_positions,
    stock_position: int = 1,
    cat_A=None, cat_B=None, cat_C=None, cat_D=None, cat_E=None,
    amounts=None,
    replicate_mode: bool = False,
    out_path: str = "tecan_operations.csv",
):
    """
    Build a pick list CSV for Tecan operations from category lists and fixed volumes.

    Parameters
    ----------
    plate_positions : list[int]
        Positions of 96-well plates on the Tecan deck.
    stock_position : int
        Labware index for the stock plate/rack.
    cat_A..cat_E : list[str]
        Category codes. Defaults match the original script.
    amounts : list[int]
        Volumes for A..E in same order.
    replicate_mode : bool
        If True, replicate every combination across additional plates in order.
    out_path : str
        Destination CSV path.

    Returns
    -------
    (csv_path, num_reaction_conditions, num_actions)
    """
    cat_A = cat_A or ["I1","I2","I3","I4","I5"]
    cat_B = cat_B or ["H1","H2","H3","H4","H5"]
    cat_C = cat_C or ["P1","P2","P3","P4"]
    cat_D = cat_D or ["C1","C2","C3","C4","C5"]
    cat_E = cat_E or ["EtOH"]
    amounts = amounts or [25, 11, 6, 20, 60]

    categories = {"A": cat_A, "B": cat_B, "C": cat_C, "D": cat_D, "E": cat_E}
    volumes = {"A": amounts[0], "B": amounts[1], "C": amounts[2], "D": amounts[3], "E": amounts[4]}
    stock = {code: idx + 1 for idx, code in enumerate(cat_A + cat_B + cat_C + cat_D + cat_E)}

    # AB C D E order mirroring the original tuple logic (A,B,C,D,E)
    combinations = [(a, b, c, d, e) for a in cat_A for b in cat_B for c in cat_C for d in cat_D for e in cat_E]

    # We map combinations onto plate positions in order
    max_wells = len(plate_positions) * 96
    if len(combinations) > max_wells:
        print(f"Warning: {len(combinations)} conditions exceed capacity {max_wells}. Truncating.")
        combinations = combinations[:max_wells]

    rows = []
    for i, combo in enumerate(combinations):
        plate_idx = i // 96
        well_num = i % 96 + 1
        assay_num = plate_positions[plate_idx]

        for j, code in enumerate(combo):
            labware = f"Labware{stock_position}"
            assay = f"Assay {assay_num}"
            pos = stock[code]
            cat_key = list(categories.keys())[j]
            volume = volumes[cat_key]
            rows.append([labware, pos, assay, well_num, volume])

        if replicate_mode:
            # Additional replicate on the next plate if available
            rep_plate_idx = plate_idx + 1
            if rep_plate_idx < len(plate_positions):
                rep_assay_num = plate_positions[rep_plate_idx]
                for j, code in enumerate(combo):
                    labware = f"Labware{stock_position}"
                    assay = f"Assay {rep_assay_num}"
                    pos = stock[code]
                    cat_key = list(categories.keys())[j]
                    volume = volumes[cat_key]
                    rows.append([labware, pos, assay, well_num, volume])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    num_conditions = len(combinations)
    num_actions = len(rows)
    return out_path, num_conditions, num_actions
