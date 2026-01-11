import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INFERENCE_DIR = os.path.join(BASE_DIR, "inference")
IMAGE_INPUT_DIR = os.path.join(INFERENCE_DIR, "input_detect_images")

METHODS = {
    "1": {
        "name": "Image Folder Detection",
        "script": "yolo_detect_input_image.py",
        "need_images": True
    },
    "2": {
        "name": "Video or USB Camera Detection",
        "script": "yolo_detect.py",
        "need_images": False
    },
    "3": {
        "name": "Share Screen Detection",
        "script": "yolo_detect_share_screen.py",
        "need_images": False
    }
}


def print_menu():
    print("\n=== YOLO Detection Launcher ===\n")
    for k, v in METHODS.items():
        print(f"{k}. {v['name']}")
    print("q. Exit\n")


def check_image_folder():

    print("\nChecking image input folder...")

    if not os.path.exists(IMAGE_INPUT_DIR):
        print("\nERROR: Folder not found.")
        print("You must create this folder:")
        print(IMAGE_INPUT_DIR)
        print("\nThen copy test images into it before running image detection.\n")
        return False

    if not os.path.isdir(IMAGE_INPUT_DIR):
        print("\nERROR: input_detect_images exists but is not a folder.")
        print("Please delete it and create a folder with this name:")
        print(IMAGE_INPUT_DIR)
        return False

    files = [
        f for f in os.listdir(IMAGE_INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if len(files) == 0:
        print("\nERROR: No images found in input_detect_images.")
        print("Copy image files here:")
        print(IMAGE_INPUT_DIR)
        print("Supported formats: jpg, jpeg, png, bmp\n")
        return False

    print(f"OK: Found {len(files)} images.")
    return True


def ask_extra_args():
    args = []

    model = input("Model path (Enter = auto latest): ").strip()
    if model:
        args += ["--model", model]

    thresh = input("Confidence threshold (default 0.5): ").strip()
    if thresh:
        args += ["--thresh", thresh]

    res = input("Resolution WxH (example 1280x720, Enter = skip): ").strip()
    if res:
        args += ["--resolution", res]

    return args


def main():

    while True:

        print_menu()
        choice = input("Select detection method: ").strip()

        if choice.lower() == "q":
            print("Exit.")
            sys.exit(0)

        if choice not in METHODS:
            print("\nInvalid option. Please select again.\n")
            continue

        method = METHODS[choice]
        script_path = os.path.join(INFERENCE_DIR, method["script"])

        if not os.path.exists(script_path):
            print("\nERROR: Detection script not found:")
            print(script_path)
            continue

        if method["need_images"]:
            ok = check_image_folder()
            if not ok:
                input("Fix the issue and press Enter to return to menu...")
                continue

        print(f"\nSelected: {method['name']}")

        use_args = input("Pass custom arguments? (y/n): ").strip().lower()

        cmd = ["python", script_path]

        if use_args == "y":
            cmd += ask_extra_args()

        print("\nRunning:")
        print(" ".join(cmd))
        print()

        subprocess.run(cmd)

        input("\nProcess finished. Press Enter to return to menu...")


if __name__ == "__main__":
    main()
