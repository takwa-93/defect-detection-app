import cv2
from pyzbar import pyzbar
import threading
import time

class BarcodeScanner:
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.last_barcode = "NONE"
        self.running = True
        self.lock = threading.Lock()
        # Start the background thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        cap = cv2.VideoCapture(self.stream_url)
        while self.running:
            # We only need the latest frame, so clear the buffer
            for _ in range(3): cap.grab()
            ret, frame = cap.retrieve()
            
            if not ret:
                time.sleep(1) # Reconnect delay
                cap = cv2.VideoCapture(self.stream_url)
                continue

            # Decode barcodes
            barcodes = pyzbar.decode(frame)
            if barcodes:
                for barcode in barcodes:
                    data = barcode.data.decode("utf-8")
                    with self.lock:
                        self.last_barcode = data
            
            time.sleep(0.1) # Small sleep to reduce CPU load

        cap.release()

    def get_latest(self):
        with self.lock:
            return self.last_barcode

    def stop(self):
        self.running = False
        self.thread.join()