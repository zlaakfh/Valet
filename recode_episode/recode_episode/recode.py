import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import os
import csv
from datetime import datetime
from ros_robot_controller_msgs.msg import BuzzerState
from sensor_msgs.msg import CompressedImage
import numpy as np
from std_msgs.msg import Int32


class DataCollectorService(Node):
    def __init__(self):
        super().__init__('data_collect_service')

        # ===== 설정 (데이터 저장 경로) =====
        self.base_data_dir = os.path.join(os.getcwd(), "collected_data")
        self.base_dir = None
        self.img_dir = None
        self.csv_file = None
        self.csv_writer = None

        self.save_hz = 10.0
        # ===================================

        self.bridge = CvBridge()
        # 여러 카메라 이미지 보관용
        self.camera_topics = [
            '/camera_0/image/compressed',
            '/camera_1/image/compressed',
            '/camera_3/image/compressed',
            '/camera_5/image/compressed',
        ]
        self.latest_images = {i: None for i in range(len(self.camera_topics))}

        self.current_v = 0.0
        self.current_w = 0.0
        self.recording_started = False  # 초기 상태: 정지
        self.parking_mode = 0  # 0:그냥, 1:주차1, 2:주차2, 3:주차3

        # 1. 서비스 서버 생성 (토픽 기반 서비스 대신 구독으로 변경)
        self.recording_sub = self.create_subscription(Int32, 'record_control', self.record_control_callback, 10)

        # 2. 토픽 구독 (각 카메라별로 구독 및 인덱스 전달)
        for idx, topic in enumerate(self.camera_topics):
            # lambda로 idx를 기본값으로 고정시켜 콜백에 전달
            self.create_subscription(CompressedImage, topic, lambda msg, i=idx: self.img_callback(msg, i), 1)
        self.cmd_sub = self.create_subscription(Twist, '/controller/cmd_vel',  self.cmd_callback, 10)

        # 토픽 발생, 녹화 시작 종료 시 부저 울림
        self.buzzer_pub = self.create_publisher(BuzzerState, 'ros_robot_controller/set_buzzer', 1)

        # 3. 타이머
        self.timer = self.create_timer(1.0 / self.save_hz, self.timer_callback)
        
        self.get_logger().info(f"🚀 데이터 수집 노드 대기 중. 저장 경로: {self.base_data_dir}")
        self.get_logger().info("서비스 요청을 보내면 녹화가 시작됩니다. (topic: /record_control)")

    def record_control_callback(self, msg):
        """정수 토픽을 받아 녹화 제어 (0:정지, 1:주차1, 2:주차2, 3:주차3, 4:일반녹화)"""
        mode = msg.data
        
        if mode == 0:
            # 정지 요청
            if self.recording_started:
                self.recording_started = False
                self.get_logger().info(">>> [명령 수신] 녹화 중지 (대기 상태)")
                
                # 녹화 중지 알림
                buzzer_msg = BuzzerState()
                buzzer_msg.freq = 2000
                buzzer_msg.on_time = 0.1
                buzzer_msg.off_time = 0.01
                buzzer_msg.repeat = 1
                self.buzzer_pub.publish(buzzer_msg)
        else:
            # 녹화 시작 요청 (모드 1, 2, 3, 4)
            if not self.recording_started:
                self.recording_started = True
                self.parking_mode = mode
                
                # 새로운 폴더 생성
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.base_dir = os.path.join(self.base_data_dir, current_time)
                self.img_dir = os.path.join(self.base_dir, "images")
                os.makedirs(self.img_dir, exist_ok=True)
                
                # CSV 파일 초기화
                self.csv_path = os.path.join(self.base_dir, "data.csv")
                self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
                self.csv_writer = csv.writer(self.csv_file)
                # 주차 모드 컬럼 추가 (p0, p1, p2)
                self.csv_writer.writerow(['timestamp', 'cam0_image', 'cam2_image', 'cam4_image', 'cam6_image', 'linear_x', 'angular_z', 'p0', 'p1', 'p2'])
                self.csv_file.flush()
                
                mode_name = {1: "주차1", 2: "주차2", 3: "주차3", 4: "일반녹화"}.get(mode, "미정의")
                self.get_logger().info(f">>> [명령 수신] 녹화 시작 ({mode_name}) - 저장 경로: {self.base_dir}")
                
                # 녹화 시작 알림
                buzzer_msg = BuzzerState()
                buzzer_msg.freq = 3000
                buzzer_msg.on_time = 0.1
                buzzer_msg.off_time = 0.01
                buzzer_msg.repeat = 1
                self.buzzer_pub.publish(buzzer_msg)

    def img_callback(self, msg, cam_idx):
        try:
            # CompressedImage -> OpenCV 이미지 디코딩
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.latest_images[cam_idx] = img
        except Exception as e:
            self.get_logger().error(f"이미지 변환 실패 (cam {cam_idx}): {e}")

    def cmd_callback(self, msg):
        # 제어값만 업데이트 (자동 시작 로직 삭제됨)
        self.current_v = msg.linear.x
        self.current_w = msg.angular.z
    
    def timer_callback(self):
        # 녹화 플래그가 꺼져있으면 저장 안 함
        if not self.recording_started:
            return

        try:
            # 4개 카메라 이미지가 모두 있을 때만 저장
            if any(self.latest_images[i] is None for i in range(len(self.camera_topics))):
                return

            # 4개 카메라 이미지를 한 번에 저장
            timestamp_str = datetime.now().strftime("%H%M%S_%f")
            filenames = {}
            
            for cam_idx in range(len(self.camera_topics)):
                img = self.latest_images[cam_idx]
                filename = f"cam{cam_idx}_img_{timestamp_str}.jpg"
                save_path = os.path.join(self.img_dir, filename)
                cv2.imwrite(save_path, img)
                filenames[cam_idx] = filename
                # 이미지 초기화 (중복 저장 방지)
                self.latest_images[cam_idx] = None

            # CSV 한 줄에 timestamp, 각 카메라 이미지파일명, 제어값, 주차 모드
            csv_row = [timestamp_str]
            for cam_idx in range(len(self.camera_topics)):
                csv_row.append(filenames.get(cam_idx, ""))
            csv_row.extend([self.current_v, self.current_w])
            
            # 원핫인코딩: p0(주차1), p1(주차2), p2(주차3)
            if self.parking_mode == 1:
                csv_row.extend([1, 0, 0])
            elif self.parking_mode == 2:
                csv_row.extend([0, 1, 0])
            elif self.parking_mode == 3:
                csv_row.extend([0, 0, 1])
            else:  # 일반 녹화 (모드 4)
                csv_row.extend([0, 0, 0])
            
            self.csv_writer.writerow(csv_row)
            self.csv_file.flush()  # CSV 버퍼 즉시 플러시

            # 실시간 로그 출력
            self.get_logger().info(f"[저장] {timestamp_str}: {filenames}")

        except Exception as e:
            self.get_logger().error(f"저장 중 에러: {e}")

    def destroy_node(self):
        if self.csv_file:
            self.csv_file.close()
        super().destroy_node()

def main():
    rclpy.init()
    node = DataCollectorService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()