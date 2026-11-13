"""Live wrist-camera window for focusing the lens by hand (2026-11-12). Read-only: the arm's bus is never opened.

The status line shows a sharpness score (variance of the Laplacian over the central half of the frame) and the best
score seen so far: turn the lens until the current value sits at the peak. Close the window to exit.

  PYTHONNOUSERSITE=1 ~/miniforge3/envs/lerobot/bin/python tools/focus_preview.py [--device 0] [--size 1280 720]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_preview import FrameGrabber, PreviewWindow, run_with_preview  # noqa: E402


def serve_http(args) -> int:
    """The same view as a multipart MJPEG stream (kill the process to stop)."""
    import cv2
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    grabber = FrameGrabber(args.device, width=args.size[0], height=args.size[1])
    best = [0.0]

    def jpeg(frame) -> bytes:
        h, w = frame.shape[:2]
        centre = cv2.cvtColor(frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4], cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(centre, cv2.CV_64F).var())
        best[0] = max(best[0], sharp)
        view = frame.copy()
        cv2.rectangle(view, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(view, f"sharpness {sharp:7.1f}   best so far {best[0]:7.1f}   (higher is sharper)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(view, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 220, 255), 1)  # the scored region
        return cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:
            pass

        def do_GET(self) -> None:
            if self.path != "/stream":
                body = b"<html><body style='margin:0;background:#111'><img src='/stream' style='width:100%'></body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            seq = -1
            try:
                while True:
                    seq, _stamp, frame = grabber.wait_for_new(seq)
                    if frame is None:
                        continue
                    data = jpeg(frame)
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode()
                                     + b"\r\n\r\n" + data + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("0.0.0.0", args.http), Handler)
    print(f"serving http://localhost:{args.http}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        grabber.close()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"),
                   help="1080p streams at only 3-6 frames/s over USB-over-IP")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--http", type=int, default=None, metavar="PORT",
                   help="serve the view as an MJPEG page on this port instead of opening a window: open http://localhost:PORT "
                        "in a Windows browser (WSLg windows are invisible when WSL was started from a Windows service session)")
    args = p.parse_args(argv)
    if args.http is not None:
        return serve_http(args)
    import cv2
    with FrameGrabber(args.device, width=args.size[0], height=args.size[1]) as grabber:
        preview = PreviewWindow("Wrist camera: focus (close the window to exit)", grabber, scale=args.scale)
        if preview.headless:
            print("no display: nothing to show")
            return 1

        def worker() -> None:
            seq, best = -1, 0.0
            while not preview.closed:
                seq, _stamp, frame = grabber.wait_for_new(seq)
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                centre = cv2.cvtColor(frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4], cv2.COLOR_BGR2GRAY)
                sharp = float(cv2.Laplacian(centre, cv2.CV_64F).var())
                best = max(best, sharp)
                preview.set_status(f"sharpness {sharp:7.1f}   best so far {best:7.1f}   ({w}x{h}; higher is sharper; close the window when done)")
                time.sleep(0.05)

        run_with_preview(worker, preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
