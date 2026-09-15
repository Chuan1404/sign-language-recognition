import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask
import cv2 as cv, json
from app.index import SignDetectionApp
from app.services.SignDetectionService import SignDetectionService
from app.utils import load_model
from config import ROOT
from src import HandDetection, PoseDetection, FusionComponent, build_gloss_list
from src.models import GPTPretrainedModel


def create_application():
    flask_app = Flask(__name__)

    hand_detection = HandDetection()
    pose_detection = PoseDetection()
    fusion = FusionComponent()

    pretrained_model = GPTPretrainedModel()
    pretrained_model.load(model_name="gpt2")

    label_dir = os.path.join(ROOT, "datasets", "annotations", "WLASL2000")

    wlasl_video_dir = os.path.join(ROOT, "datasets", "raw", "WLASL", "videos")

    model_path = os.path.join(ROOT, "outputs", "models", "contest_2000_v4_2.pt",)

    with open(os.path.join(label_dir, "gloss2idx.json"), "r",) as f:
        gloss2idx = json.load(f)

    idx2gloss = {v: k for k, v in gloss2idx.items()}

    gloss_list = build_gloss_list(label_dir=label_dir, video_dir=wlasl_video_dir)
    model = load_model(model_path=model_path,num_classes=len(gloss2idx))

    service = SignDetectionService(
        hand_detection=hand_detection,
        pose_detection=pose_detection,
        fusion=fusion,
        model=model,
        pretrained_model=pretrained_model,
        idx2gloss=idx2gloss,
    )

    camera = cv.VideoCapture(0, cv.CAP_DSHOW)
    application = SignDetectionApp(
        flask_app=flask_app,
        service=service,
        camera=camera,
        gloss_list=gloss_list,
        wlasl_video_dir=wlasl_video_dir,
    )

    return application.app

app = create_application()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False,)