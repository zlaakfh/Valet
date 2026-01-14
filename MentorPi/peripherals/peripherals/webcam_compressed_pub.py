import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

class MultiCompressedWebcamPublisher(Node):
    def __init__(self):
        super().__init__('multi_compressed_webcam_publisher')
        
        # v4l2-ctl --list-devices 로 확인 필요
        # self.camera_indices = [0, 2, 4, 6]
        self.camera_indices = [0, 1, 3, 5]
        self.cameras = []
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        
        self.callback_group = ReentrantCallbackGroup()

        for idx in self.camera_indices:
            cap = cv2.VideoCapture(idx)
            
            if cap.isOpened():
                # MJPEG 설정 (대역폭 절약)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 10)
                
                # 버퍼 사이즈를 1로 줄여서 지연(Lag) 방지 (backend에 따라 지원 여부 다름)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                topic_name = f'/camera_{idx}/image/compressed'
                publisher = self.create_publisher(CompressedImage, topic_name, 1)
                
                # [수정 3] 카메라마다 개별 타이머 생성 & 콜백 그룹 할당
                # 10FPS = 약 0.1초 주기
                timer = self.create_timer(
                    0.1, 
                    lambda cap=cap, pub=publisher, c_id=idx: self.camera_callback(cap, pub, c_id),
                    callback_group=self.callback_group
                )
                
                self.cameras.append({'cap': cap, 'timer': timer})
                self.get_logger().info(f'Camera {idx} initialized (MJPEG).')
            else:
                self.get_logger().error(f'Could not open camera {idx}')

    def camera_callback(self, cap, publisher, camera_id):
        # 개별 스레드에서 실행됨 (하나가 느려져도 다른 카메라에 영향 없음)
        ret, frame = cap.read()
        if ret:
            success, encoded_image = cv2.imencode('.jpg', frame, self.encode_param)
            if success:
                msg = CompressedImage()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = f"camera_frame_{camera_id}"
                msg.format = "jpeg"
                msg.data = np.array(encoded_image).tobytes()
                publisher.publish(msg)

    def close_cameras(self):
        for cam in self.cameras:
            if cam['cap'].isOpened():
                cam['cap'].release()

def main(args=None):
    rclpy.init(args=args)
    node = MultiCompressedWebcamPublisher()
    

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.close_cameras()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


'''
# Subscriber 콜백 함수 예시
def listener_callback(self, msg):
    # 1. 바이트 데이터를 numpy 배열로 변환
    np_arr = np.frombuffer(msg.data, np.uint8)
    
    # 2. 이미지 디코딩 (압축 해제)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    cv2.imshow("Received", image_np)
    cv2.waitKey(1)
'''