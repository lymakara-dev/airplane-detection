# Aircraft Detection with YOLO

This project provides a complete local workflow for labeling, training, and running a YOLO-based object detection model using datasets annotated with Label Studio.

The workflow covers:

- Image labeling with Label Studio (local, Conda)
- Dataset preparation and splitting
- YOLO training
- Real-time inference on images, videos, and USB cameras

---

## Project Structure

    aircraft-detection/
    ├─ dataset/
    │ ├─ images/ # raw images copied from Label Studio storage
    │ ├─ labels/ # raw YOLO labels from Label Studio export
    │ ├─ train/
    │ │ ├─ images/
    │ │ └─ labels/
    │ ├─ val/
    │ │ ├─ images/
    │ │ └─ labels/
    │ ├─ classes.txt # class names from Label Studio
    │ └─ data.yaml # YOLO dataset configuration
    │
    ├─ scripts/
    │ ├─ split_dataset.py # train/val split script
    │ └─ create_data_yaml.py # auto-generate data.yaml
    │
    ├─ inference/
    │ └─ yolo_detect.py # inference script (image, video, webcam)
    │
    ├─ models/ # final trained models (manually copied)
    ├─ runs/ # YOLO training outputs (auto-generated)
    │
    ├─ yolo11n.pt # pretrained base model
    ├─ yolo11s.pt # pretrained base model
    ├─ requirements.txt
    ├─ .gitignore
    └─ README.md

## Setup Environment

Create and activate conda environment:

    conda create -n yolo-env python=3.12
    conda activate yolo-env

Install dependencies:

    pip install -r requirements.txt

Verify GPU:

    python -c "import torch; print(torch.cuda.is_available())"

## Dataset Preparation

1. Export YOLO labels from Label Studio

Start label-studio:

    label-studio start

Go to: `http://localhost:8080` and do the labeling and export zip file

2. Copy images into dataset/images
3. Place labels into dataset/labels
4. Run dataset split:

   python scripts/split_dataset.py --datapath dataset --train_pct 0.9

5. Generate data.yaml:

   python scripts/create_data_yaml.py

## Training

    yolo detect train model=yolo11s.pt data=dataset/data.yaml epochs=60 imgsz=640

Training results are saved to:

`runs/detect/train/weights/best.pt`

## Model Management

After training, copy the final model into models/:

    copy runs\detect\train\weights\best.pt models\aircraft.pt

## Inference

Webcam:

    python inference/yolo_detect.py --model models/aircraft.pt --source usb0 --resolution 1280x720

Video file:

    python inference/yolo_detect.py --model models/aircraft.pt --source video.mp4

## Notes

- Use numeric camera index on Windows

## License

For academic and research use.
