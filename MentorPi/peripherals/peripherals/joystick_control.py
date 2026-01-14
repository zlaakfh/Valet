#!/usr/bin/env python3
# encoding: utf-8
import os
import sys
import struct
import math
import time
import threading
import rclpy
from enum import Enum
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool
from std_msgs.msg import Int32

from ros_robot_controller_msgs.msg import BuzzerState, SetPWMServoState, PWMServoState


# --- Helper Function ---
def val_map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# --- Constants ---
AXES_MAP = ['lx', 'ly', 'rx', 'ry', 'r2', 'l2', 'hat_x', 'hat_y']
# BUTTON_MAP = ['cross', 'circle', '', 'square', 'triangle', '', 'l1', 'r1', 'l2', 'r2', 'select', 'start', '', 'l3', 'r3', '', 'hat_xl', 'hat_xr', 'hat_yu', 'hat_yd', '']
BUTTON_MAP = ['Y', 'B', 'A', 'X', 'l1', 'r1', 'l2', 'r2', 'select', 'start', 'l3', 'r3','mode']

class ButtonState(Enum):
    Normal = 0
    Pressed = 1
    Holding = 2
    Released = 3

class DirectJoystickController(Node):
    def __init__(self, name):
        super().__init__(name)
        
        # --- ROS 2 파라미터 ---
        self.declare_parameter('max_linear', 0.7)
        self.declare_parameter('max_angular', 3.0)
        self.declare_parameter('cam_x', 1500)
        self.declare_parameter('cam_y', 1500)
        self.declare_parameter('disable_servo_control', True)

        self.max_linear = self.get_parameter('max_linear').value
        self.max_angular = self.get_parameter('max_angular').value
        self.cam_x = self.get_parameter('cam_x').value
        self.cam_y = self.get_parameter('cam_y').value
        self.disable_servo_control = self.get_parameter('disable_servo_control').value
        self.machine = os.environ.get('MACHINE_TYPE', 'MentorPi_Acker')

        self.get_logger().info(f'Machine: {self.machine}, Max Lin: {self.max_linear}')

        # --- Publishers ---
        self.mecanum_pub = self.create_publisher(Twist, 'controller/cmd_vel', 10)
        
        
        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 10)
        self.buzzer_pub = self.create_publisher(BuzzerState, 'ros_robot_controller/set_buzzer', 10)

        # self.recording = self.create_client(SetBool, 'record_control')
        self.recording = self.create_publisher(Int32, 'record_control', 2)

        # --- 상태 변수 ---
        self.min_value = 0.1
        self.last_axes = dict(zip(AXES_MAP, [0.0] * len(AXES_MAP)))
        self.last_buttons = dict(zip(BUTTON_MAP, [0.0] * len(BUTTON_MAP)))

        # Raw 데이터 저장을 위한 리스트 (Linux Joystick Standard)
        # Axis 8개, Button 15개 정도를 가정하고 넉넉히 잡음
        self.raw_axes = [0.0] * 10
        self.raw_buttons = [0] * 20

        # --- 쓰레드 시작 ---
        # 파일 읽기는 Blocking이므로 메인 ROS 스핀과 분리해야 함
        self.running = True
        self.read_thread = threading.Thread(target=self.read_joystick_loop)
        self.read_thread.daemon = True
        self.read_thread.start()

        self.get_logger().info('Direct Joystick Controller Started (Raw Mode)')

    def read_joystick_loop(self):
        """ /dev/input/js0를 직접 읽는 쓰레드 """
        device_path = "/dev/input/js0"
        
        if not os.path.exists(device_path):
            self.get_logger().error(f"Device {device_path} not found. Check connection.")
            return

        try:
            with open(device_path, "rb") as js_file:
                while self.running and rclpy.ok():
                    # 8바이트 이벤트 읽기
                    event_data = js_file.read(8)
                    if not event_data: break

                    time_ms, value, type_, number = struct.unpack("IhBB", event_data)
                    
                    # 초기화 이벤트(0x80) 마스킹
                    real_type = type_ & ~0x80
                    updated = False

                    if real_type == 1: # Button
                        if number < len(self.raw_buttons):
                            self.raw_buttons[number] = 1 if value else 0
                            updated = True
                    
                    elif real_type == 2: # Axis
                        if number < len(self.raw_axes):
                            # -32767 ~ 32767 값을 -1.0 ~ 1.0 으로 정규화
                            norm_val = value / 32767.0
                            # 미세 노이즈 제거 (아주 작은 값 0 처리)
                            if abs(norm_val) < 0.05: norm_val = 0.0
                            self.raw_axes[number] = norm_val
                            updated = True

                    # 데이터 변경이 있을 때 제어 로직 실행
                    if updated:
                        self.process_control_logic()

        except PermissionError:
            self.get_logger().error(f"Permission denied accessing {device_path}. Run with sudo.")
        except Exception as e:
            self.get_logger().error(f"Joystick read error: {e}")

    def process_control_logic(self):
        """
        raw_axes와 raw_buttons 상태를 기반으로 ROS 메시지 생성 및 발행
        (기존 joy_callback 로직과 통합)
        """
        # 1. 딕셔너리 매핑 (Joy 메시지 구조 모방)
        current_axes_dict = dict(zip(AXES_MAP, self.raw_axes[:len(AXES_MAP)]))
        
        # Hat(D-pad) 처리: Axis 6, 7번을 Hat 버튼으로 변환
        hat_x = current_axes_dict.get('hat_x', 0)
        hat_y = current_axes_dict.get('hat_y', 0)
        
        hat_xl, hat_xr = (1 if hat_x > 0.5 else 0), (1 if hat_x < -0.5 else 0)
        hat_yu, hat_yd = (1 if hat_y > 0.5 else 0), (1 if hat_y < -0.5 else 0)
        
        # 버튼 리스트 구성: 물리 버튼 + Hat 버튼 + 더미
        current_buttons_list = self.raw_buttons[:12] # 표준 버튼들
        current_buttons_list.extend([hat_xl, hat_xr, hat_yu, hat_yd, 0])
        
        current_buttons_dict = dict(zip(BUTTON_MAP, current_buttons_list))

        # 2. Axis 처리 (Twist 발행)
        axes_changed = False
        for key in AXES_MAP:
            if self.last_axes[key] != current_axes_dict.get(key, 0):
                axes_changed = True
                break
        
        if axes_changed:
            self.axes_callback(current_axes_dict)

        # 3. Button 처리 (Callback 호출)
        for key, val in current_buttons_dict.items():
            if not key: continue

            last_val = self.last_buttons.get(key, 0)
            if val != last_val:
                new_state = ButtonState.Pressed if val > 0 else ButtonState.Released
            else:
                new_state = ButtonState.Holding if val > 0 else ButtonState.Normal
            
            if new_state != ButtonState.Normal:
                callback_name = f"{key}_callback"
                if hasattr(self, callback_name):
                    try:
                        getattr(self, callback_name)(new_state)
                    except Exception as e:
                        self.get_logger().error(f"Error in {callback_name}: {e}")

        # 상태 업데이트
        self.last_axes = current_axes_dict
        self.last_buttons = current_buttons_dict

    def axes_callback(self, axes):
        twist = Twist()
        # Deadzone 체크
        lx = axes['lx'] if abs(axes['lx']) > self.min_value else 0.0
        ly = axes['ly'] if abs(axes['ly']) > self.min_value else 0.0
        rx = axes['rx'] if abs(axes['rx']) > self.min_value else 0.0
        ry = axes['ry'] if abs(axes['ry']) > self.min_value else 0.0

        # if self.machine == 'MentorPi_Mecanum':
        #     twist.linear.y = val_map(lx, -1, 1, -self.max_linear, self.max_linear) 
        #     twist.linear.x = val_map(ly, -1, 1, -self.max_linear, self.max_linear)
        #     twist.angular.z = val_map(rx, -1, 1, -self.max_angular, self.max_angular)
        # elif self.machine == 'JetRover_Tank':
        #     twist.linear.x = val_map(ly, -1, 1, -self.max_linear, self.max_linear)
        #     twist.angular.z = val_map(rx, -1, 1, -self.max_angular, self.max_angular)
        # elif self.machine == 'MentorPi_Acker':
        #     twist.linear.x = val_map(ly, -1, 1, -self.max_linear, self.max_linear)
        #     steering_angle = val_map(rx, -1, 1, -math.radians(45), math.radians(45))
        
        ### MentorPi_Acker
        twist.linear.x = -val_map(ly, -1, 1, -self.max_linear, self.max_linear)
        steering_angle = -val_map(rx, -1, 1, -math.radians(45), math.radians(45))            
        
        self.get_logger().info(f'{steering_angle}, {twist.linear.x}')
        # 서보 제어 로직
        if self.disable_servo_control:
            servo_state = PWMServoState()
            servo_state.id = [3]
            
            if steering_angle == 0:
                twist.angular.z = 0.0
                servo_state.position = [1500]
            else:
                try:
                    R = 0.145 / math.tan(steering_angle)
                    twist.angular.z = float(twist.linear.x / R)
                except ZeroDivisionError:
                    twist.angular.z = 0.0
                servo_state.position = [1500 + int(math.degrees(steering_angle) / 180 * 2000)]
            
            data = SetPWMServoState()
            data.state = [servo_state]
            data.duration = 0.02
            self.servo_state_pub.publish(data)

        self.mecanum_pub.publish(twist)

    # --- Button Callbacks ---
    def select_callback(self, new_state): 
        self.get_logger().info(f'Select: {new_state}')
        
        msg = Int32()
        msg.data = 4
        self.recording.publish(msg)

    def start_callback(self, new_state):
        if new_state == ButtonState.Pressed:
            msg = BuzzerState()
            msg.freq = 2500
            msg.on_time = 0.05
            msg.off_time = 0.01
            msg.repeat = 1
            self.buzzer_pub.publish(msg)
            self.get_logger().info('Buzzer Beep!')

    def mode_callback(self, new_state): self.get_logger().info(f'mode: {new_state}')

    # camara up
    def Y_callback(self, new_state): 
        self.get_logger().info(f'Y: {new_state} {self.cam_y}')
        
        msg = Int32()
        msg.data = 2
        self.recording.publish(msg)


    def B_callback(self, new_state): 
        self.get_logger().info(f'B: {new_state} {self.cam_x}')
        
        msg = Int32()
        msg.data = 3
        self.recording.publish(msg)


    def A_callback(self, new_state): 
        self.get_logger().info('A Pressed: Stop Recording Request')
                    
        msg = Int32()
        msg.data = 0
        self.recording.publish(msg)


    def X_callback(self, new_state): 
        # recode msg send
        self.get_logger().info('X Pressed: Start Recording Request')
        
        msg = Int32()
        msg.data = 1
        self.recording.publish(msg)

    def l1_callback(self, new_state): 
        self.get_logger().info(f'L1: {new_state}')
        
    def r1_callback(self, new_state): self.get_logger().info(f'R1: {new_state}')

    def l2_callback(self, new_state):
        if self.max_linear - 0.05 > 0:
            self.max_linear -= 0.05
        self.get_logger().info(f'L2: {new_state}, {self.max_linear}')
        

    def r2_callback(self, new_state):
        if self.max_linear + 0.05 <= 1:
            self.max_linear += 0.05
        self.get_logger().info(f'R2: {new_state}, {self.max_linear}')

    def l3_callback(self, new_state): self.get_logger().info(f'L3: {new_state}')
    def r3_callback(self, new_state): self.get_logger().info(f'R3: {new_state}')
    
    def triangle_callback(self, new_state): self.get_logger().info(f'Triangle: {new_state}')
    def cross_callback(self, new_state): self.get_logger().info(f'Cross: {new_state}')
    def square_callback(self, new_state): self.get_logger().info(f'Square: {new_state}')
    def circle_callback(self, new_state): self.get_logger().info(f'Circle: {new_state}')

def main(args=None):
    rclpy.init(args=args)
    node = DirectJoystickController('direct_joystick_ctrl')
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by user')
        node.running = False
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()