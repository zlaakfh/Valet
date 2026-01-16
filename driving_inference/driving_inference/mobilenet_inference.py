import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from .model_pytorch import MobileNet

import torch
import numpy as np
import cv2


class MobileNetInferenceNode(Node):
    def __init__(self):
        super().__init__('mobilenet_inference_node')

        self.declare_parameter('CROP_HEIGHT', 100)
        self.declare_parameter('LEANER_GAIN', 0.8)
        self.declare_parameter('STEERING_GAIN', 1.0)
        self.declare_parameter('BACKBONE', 'v2')  # v2 또는 v3s
        self.declare_parameter('MODEL_PATH', 'mobilenet_v2_one_cycle_lr_batch_128_epoch_100_lr_001_model.pth')
        self.declare_parameter('input_video', '/camera_2/image/compressed')

        self.CROP_HEIGHT = self.get_parameter('CROP_HEIGHT').get_parameter_value().integer_value
        self.LEANER_GAIN = self.get_parameter('LEANER_GAIN').get_parameter_value().double_value
        self.STEERING_GAIN = self.get_parameter('STEERING_GAIN').get_parameter_value().double_value
        self.BACKBONE = self.get_parameter('BACKBONE').get_parameter_value().string_value
        self.MODEL_PATH = self.get_parameter('MODEL_PATH').get_parameter_value().string_value
        self.input_video = self.get_parameter('input_video').get_parameter_value().string_value

        # 구독자: 카메라 이미지
        self.image_sub = self.create_subscription(
            CompressedImage,
            self.input_video,
            self.image_callback,
            1
        )

        # 발행자: 제어 명령
        self.cmd_pub = self.create_publisher(
            Twist,
            '/controller/cmd_vel',
            10
        )

        # 모델 로드 (여기서는 더미 모델 사용)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()

    def load_model(self):
        # (1) 모델 껍데기(아키텍처) 생성
        model = MobileNet(
            backbone=self.BACKBONE,
            pretrained=False,         # ✅ YUV + [-1,1]이면 일단 False 추천
            freeze_backbone=False,
            input_norm="minus1_1",    # ✅ 지금 PilotNet과 동일
            dropout_p=0.2,
            out_dim=2
        )
            
        # (2) 가중치 파일 로드
        # map_location은 GPU에서 학습한 모델을 CPU에서 돌리거나 할 때 필수
        checkpoint = torch.load(self.MODEL_PATH, map_location=self.device)
        
        # (3) 가중치 입히기
        model.load_state_dict(checkpoint)
        
        # (4) 추론 모드 전환 및 디바이스 이동
        model.to(self.device)
        model.eval()

        self.get_logger().info('Model loaded successfully.')
        return model

    def image_callback(self, msg):
        # 이미지 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 모델 추론 수행
        v, w = self.run_inference(image)

        # Twist 메시지 생성 및 발행
        twist_msg = Twist()
        twist_msg.linear.x = v
        twist_msg.angular.z = w
        self.cmd_pub.publish(twist_msg)

    def run_inference(self, image):
        
        input_tensor = self.preprocess(image)

        # [중요] 추론 시에는 불필요한 그라디언트 계산을 꺼야 속도가 빠르고 메모리를 아낍니다.
        with torch.no_grad():
            output = self.model(input_tensor)
        
        v = output[0, 0].item()  # 첫 번째 값 (선형 속도)
        w = output[0, 1].item()  # 두 번째 값 (각속도)
        self.get_logger().info(f'Predicted v: {v:.4f}, w: {w:.4f}')
        
        return v, w

    def preprocess(self, image):
        image = image[self.CROP_HEIGHT:480, :]
        image = cv2.resize(image, (200, 66)) # Resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) # YUV 변환

        image = image.transpose((2, 0, 1))
        
        # (B) NumPy -> Tensor 변환
        input_tensor = torch.from_numpy(image)
        
        # (C) 데이터 타입 변환 (uint8 -> float32)
        # 모델 내부에서 .float()를 하더라도 밖에서 해주는 것이 명확합니다.
        input_tensor = input_tensor.float()
        
        # (D) 배치 차원 추가: (C, H, W) -> (1, C, H, W)
        # 결과: (3, 66, 200) -> (1, 3, 66, 200)
        input_tensor = input_tensor.unsqueeze(0)
        
        # (E) GPU로 이동
        input_tensor = input_tensor.to(self.device)
        return input_tensor

def main(args=None):
    rclpy.init(args=args)
    node = MobileNetInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()