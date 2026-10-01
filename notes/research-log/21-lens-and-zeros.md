# 21. What lens does the real camera have, and are the joints right now? (Sep 8, 2026)

**TL;DR:** A checkerboard on my phone showed the lens is 44 degrees, not the 72
the sim used, and the arm's joint zeros needed one more correction.

**Problem.** Even with the new joint map, the sim still put the square below the
frame and 2.6x too small.

*How we know:* redoing entry 19's overlay gave a 218 px square against the real
574 px. I didn't know the lens, the mount position, or whether the zeros were
finally right.

**Experiment.** Show a checkerboard on my iPhone to the camera: 38 views to fit
the lens (OpenCV calibration), then 9 frames with the phone lying flat on the
mat while the arm moves, comparing where the board appears with where the arm
model says the camera is. One side photo of the resting arm to break a tie.

*What we hope to learn:* the lens's field of view and distortion, where the
camera sits, and whether the joint zeros are right.

**Outcome.** Field of view 44 degrees with strong barrel distortion (the
published 103 for this camera is wrong for this lens). The mount is the official
part, placed within a centimetre:

- A flat board doesn't move, so if the arm model is right, every view should put
  it in the same place. With the old zeros it scattered by 61 mm; with corrected
  ones it agreed to about 21-23 mm. Like triangulating a landmark from several spots.
- The board couldn't tell a shoulder error from an elbow error. The side photo
  could, and settled the final zeros.

The script re-certified 15/15, and I signed off the camera review.

**Takeaway:** a still object seen from many poses is a free calibration rig. When
your data can't separate two explanations, get one more independent measurement.
