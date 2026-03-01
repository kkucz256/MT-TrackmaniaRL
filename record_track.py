import time
import csv
from pipeline import TrackmaniaPipeline

def record():
    pipe = TrackmaniaPipeline()
    points = []
    print("--- TRYB NAGRYWANIA TRASY ---")
    print("Masz 5 sekund na wejście do gry i start.")
    time.sleep(5)
    
    try:
        while True:
            _, tele = pipe.get_state()
            if tele and 'pos_x' in tele:
                pos = (tele['pos_x'], tele['pos_y'], tele['pos_z'])
                if not points or np.linalg.norm(np.array(pos) - np.array(points[-1])) > 1.0:
                    points.append(pos)
                    print(f"Zapisano punkt {len(points)}: {pos}", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print(f"\nZakończono nagrywanie. Zapisano {len(points)} punktów.")
        with open("track_points.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(points)
        print("Dane zapisane do track_points.csv")
    finally:
        pipe.stop()

if __name__ == "__main__":
    import numpy as np
    record()