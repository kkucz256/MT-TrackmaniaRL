import socket
import json
import threading
import time
import cv2
import bettercam
import win32gui

def get_trackmania_window():
    hwnd = win32gui.FindWindow(None, "Trackmania")
    if not hwnd:
        for window_name in ["TrackMania", "Trackmania 2020", "tm2020", "TM2020"]:
            hwnd = win32gui.FindWindow(None, window_name)
            if hwnd:
                break
        
        if not hwnd:
            return None
    
    rect = win32gui.GetWindowRect(hwnd)
    return rect

class TrackmaniaPipeline:
    _instance_count = 0
    _lock = threading.Lock()
    
    def __init__(self, tcp_ip="0.0.0.0", tcp_port=9000):
        
        with TrackmaniaPipeline._lock:
            TrackmaniaPipeline._instance_count += 1
            instance_id = TrackmaniaPipeline._instance_count
        
        self.running = True
        self.latest_telemetry = {}
        self.instance_id = instance_id
        
        window_rect = get_trackmania_window()
        
        if window_rect:
            left, top, right, bottom = window_rect
            
            if left < 0 or top < 0:
                self.camera = bettercam.create(output_color="BGR")
            else:
                self.camera = bettercam.create(region=window_rect, output_color="BGR")
        else:
            self.camera = bettercam.create(output_color="BGR")
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.sock.bind((tcp_ip, tcp_port))
                break
            except OSError as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise
        
        self.sock.listen(1)
        
        self.conn = None
        self.addr = None
        self.sock.settimeout(1.0)
        
        self.thread = threading.Thread(target=self._tcp_worker, daemon=True)
        self.thread.start()

    def _tcp_worker(self):
        buffer = ""
        
        while self.running:
            if self.conn is None:
                try:
                    self.conn, self.addr = self.sock.accept()
                    self.conn.settimeout(5.0)
                    print(f"\n[Pipeline] Caught telemetry from {self.addr}\n")
                except socket.timeout:
                    continue
                except Exception:
                    time.sleep(0.5)
                    continue
            
            try:
                data = self.conn.recv(4096)
                if not data:
                    self.conn.close()
                    self.conn = None
                    continue
                
                buffer += data.decode('utf-8', errors='ignore')
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if line:
                        try:
                            self.latest_telemetry = json.loads(line)
                        except json.JSONDecodeError as e:
                            print(f"[JSON ERROR]: Data Rejected! Cause: {e} | Raw: {line}")
                        except UnicodeDecodeError:
                            continue
            except socket.timeout:
                continue
            except (OSError, socket.error):
                if self.conn:
                    self.conn.close()
                self.conn = None
                time.sleep(0.2)
            except Exception:
                if self.running:
                    time.sleep(0.2)

    def get_state(self):
        frame = self.camera.grab()
        
        if frame is not None:
            height, width, _ = frame.shape
            
            crop_top = int(height * 0.40)
            crop_bottom = int(height * 0.85)
            
            frame_cropped = frame[crop_top:crop_bottom, :]
            frame_resized = cv2.resize(frame_cropped, (128, 128))
            
            return frame_resized, self.latest_telemetry
        
        return None, self.latest_telemetry

    def stop(self):
        self.running = False
        if self.conn:
            self.conn.close()
        self.sock.close()
        self.camera.release()

if __name__ == "__main__":
    pipeline = TrackmaniaPipeline()
    time.sleep(1)
    
    last_time = time.time()
    frames = 0
    
    try:
        while True:
            frame, tele = pipeline.get_state()
            
            if frame is not None:
                cv2.imshow("CNN Vision (128x128 Cropped)", frame)
                
                frames += 1
                if time.time() - last_time >= 1.0:
                    print(f"Loop: {frames} FPS | Game Telemetry: {tele}")
                    frames = 0
                    last_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()