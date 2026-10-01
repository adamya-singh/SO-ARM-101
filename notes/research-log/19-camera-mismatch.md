# 19. Does the sim camera see what the real camera sees? (Sep 6, 2026)

**TL;DR:** No: the sim showed the square 4x smaller and 300 pixels lower than
the real camera did.

**Problem.** I had never compared the sim's wrist camera with the real one.
Training on wrong renders teaches the policy a view the real arm never sees.

*How we know:* I didn't, and that was the problem. No image from the real camera
had ever been checked against the sim.

**Experiment.** Put the real arm and the sim arm in the same recorded pose,
render the sim view at the real camera's resolution, and overlay the sim square
on the real photo.

*What we hope to learn:* if they match, a quick image review clears training.
If not, the camera needs calibrating first.

**Outcome.** Way off. The real square was 574 px wide near the top of the
frame; the sim square was 139 px wide, about 300 px lower. Only the horizontal
centring agreed.

- My explanations that night were guesses, and both were wrong: I guessed the
  real lens was about 36 degrees (it's 44) and that the mount was a different
  part (it's the official one). Part of the gap was actually the joint map
  (entry 20).

I blocked training and changed the first gate from "review the images" to
"calibrate the camera."

**Takeaway:** compare sim and real images side by side, at the same pose, before
you train on anything. And don't write a guess down as if it were the
explanation.
