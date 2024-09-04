from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
from demo import load_checkpoints
from demo import AnimationMaker
from skimage import img_as_ubyte

app = FastAPI()

source_image = cv2.imread('data/monariza.png')
if source_image is None:
    print("画像が正しく読み込まれていません")
else:
    print("画像が正常に読み込まれました")

source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
source_image = cv2.resize(source_image, (256, 256))
source_image = source_image.astype(np.float32) / 255.0

generator, kp_detector = load_checkpoints(
    config_path='config/vox-256.yaml',
    checkpoint_path='vox-cpk.pth.tar')

animation_maker = AnimationMaker(generator, kp_detector, source_image)


@app.websocket("/ws")
async def detect_objects(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_bytes()
    
            nparr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                print("受信したデータのデコqードに失敗しました")
                continue

            driving_frame = cv2.resize(image, (256, 256))
            driving_frame = cv2.cvtColor(driving_frame, cv2.COLOR_BGR2RGB)
            driving_frame = driving_frame.astype(np.float32) / 255.0

            result_frame = animation_maker.make_animation(driving_frame)
            result_image = img_as_ubyte(result_frame)
            result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

            _, buffer = cv2.imencode('.jpg', result_image)
            await websocket.send_bytes(buffer.tobytes())
        
        except WebSocketDisconnect:
            break