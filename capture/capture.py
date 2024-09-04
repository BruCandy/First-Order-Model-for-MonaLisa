import asyncio
import websockets
import cv2
import numpy as np


async def send_images():
    # WebSocketサーバーのURI
    uri = "ws://localhost:8000/ws"

    # カメラのキャプチャを開始
    cap = cv2.VideoCapture(0)

    async with websockets.connect(uri) as websocket:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # フレームをJPEG形式でエンコード
            _, img_encoded = cv2.imencode('.jpg', frame)

            # WebSocketでサーバーに画像を送信
            await websocket.send(img_encoded.tobytes())

            # サーバーからの応答を受信（物体検出結果画像）
            try:
                data = await websocket.recv()
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"Connection closed with error: {e}")
                break

            # 受信したバイトデータを画像にデコード
            nparr = np.frombuffer(data, np.uint8)
            result_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 結果画像を表示
            cv2.imshow("generated", result_image)
            cv2.imshow("original", frame)

            # 'q'キーを押すとループを終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(send_images())