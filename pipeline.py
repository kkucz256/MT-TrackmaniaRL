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
        print("BŁĄD: Nie znaleziono okna Trackmanii! Upewnij się, że gra jest włączona.")
        return None
    return win32gui.GetWindowRect(hwnd)

class TrackmaniaPipeline:
    def __init__(self, tcp_ip="0.0.0.0", tcp_port=9000):
        self.running = True
        self.latest_telemetry = {}
        
        window_rect = get_trackmania_window()
        if window_rect:
            left, top, right, bottom = window_rect
            print(f"Znaleziono okno gry. Region przechwytywania: {window_rect}")
            
            if left < 0 or top < 0:
                print("UWAGA: Okno poza głównym ekranem! Przechwytywanie całego ekranu...")
                self.camera = bettercam.create(output_color="BGR")
            else:
                self.camera = bettercam.create(region=window_rect, output_color="BGR")
        else:
            self.camera = bettercam.create(output_color="BGR") 
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((tcp_ip, tcp_port))
        self.sock.listen(1)
        
        print(f"--- Nasłuchuję TCP na porcie {tcp_port}. Przeładuj plugin w OpenPlanet! ---")
        self.conn, self.addr = self.sock.accept()
        self.conn.settimeout(5.0)
        print(f"POŁĄCZONO Z GRĄ: {self.addr}")

        self.thread = threading.Thread(target=self._tcp_worker, daemon=True)
        self.thread.start()

    def _tcp_worker(self):
        buffer = ""
        while self.running:
            try:
                try:
                    data = self.conn.recv(4096)
                except socket.timeout:
                    time.sleep(0.5)
                    continue
                
                if not data:
                    print("Zerwano połączenie TCP.")
                    break
                
                buffer += data.decode('utf-8')
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if line:
                        self.latest_telemetry = json.loads(line)
            except Exception as e:
                print(f"Błąd w wątku TCP: {e}")
                time.sleep(0.5)

    def get_state(self):
        frame = self.camera.grab()
        
        if frame is not None:
            height, width, _ = frame.shape
            
            crop_top = int(height * 0.40)     # Ucinamy górne 40% ekranu
            crop_bottom = int(height * 0.85)  # Ucinamy dolne 15% ekranu
            
            frame_cropped = frame[crop_top:crop_bottom, :]
            
            frame_resized = cv2.resize(frame_cropped, (128, 128))
            
            return frame_resized, self.latest_telemetry
            
        return None, self.latest_telemetry

    def stop(self):
        self.running = False
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
                    print(f"Pętla: {frames} FPS | Telemetria z gry: {tele}")
                    frames = 0
                    last_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()