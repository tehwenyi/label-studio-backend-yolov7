# Label Studio with YOLOv7 Backend
Label Studio ML Backend using YOLOv7 (https://github.com/yhsmiley/yolov7).

Currently only able to predict. Training has not been incorporated yet.

## Quick Start

### Step 1: Set Up YOLOv7 Backend
1. Git clone this repository.
```bash
git clone https://github.com/tehwenyi/label-studio-backend-yolov7.git
cd label-studio-backend-yolov7/
```

2. Copy your **YOLOv7 weights** (`.pt` file) into the `weights` folder and rename it to `weights.pt`.
- Please make sure that your weights have been reparameterised and the state dictionary has been saved. Refer to this [link](https://github.com/DinoHub/yolov7_pipeline/tree/main#to-convert-weights-for-use-in-inference-branch) to find out how to reparameterise and save the state dictionary.

3. Copy your **YOLOv7 deploy cfg** (eg. `deploy/cfg/yolov7.yaml`) into the `cfg` folder and rename it to `cfg.yaml`.

4. Start running the backend with Docker Compose.
```bash
docker compose up
```

### Step 2: Start Label Studio
Follow the instructions on https://github.com/tehwenyi/label-studio to set up with **Docker Compose**.

### Step 3: Start the ML Backend on your Project
1. In the Label Studio UI, open the project that you want to use with your ML backend.
2. Make sure that the **labels** you input in the project **correspond exactly to your model's trained labels**, otherwise you might get a `No Label` annotation after prediction.
3. Click **Settings > Machine Learning**.
4. Click **Add Model**.
5. Type a Title for the model and provide type **http://server:9090** for the URL for the ML backend.
6. (Optional) Type a description.
7. (Optional) Select **Retrieve predictions when loading a task automatically** to allow predictions on all images.
8. Click Validate and Save.
Instructions adapted from [Label Studio Documentation](https://labelstud.io/guide/ml.html#Add-an-ML-backend-to-Label-Studio)

## Notes
- For any changes to the YOLOv7 inference parameters, edit the parameters in **Line 21** of `backend.py`
- The Docker containers are added to the same network `label-studio`.
- The YOLOv7 backend uses Python 3.9.
- The YOLOv7 backend uses port `9090` while label studio typically uses `8080`.
- To find out how to create your own ML backend, refer to [heartexlabs/label-studio-ml-backend](https://github.com/heartexlabs/label-studio-ml-backend)# label-studio-backend-yolov7
