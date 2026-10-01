# 15. Can it do this from pixels? (Aug 6, 2026)

**TL;DR:** The first vision policy was bad as expected (0-3/15), and blanking the
camera barely changed it, so it was mostly ignoring the image.

**Problem.** Every policy so far read the cube's exact position straight from
the simulator. A real arm doesn't get that.

*How we know:* the cube position was one of the policy's inputs. I didn't know
how much worse it would be without it.

**Experiment.** Replace the cube position with the wrist camera image (256×256)
through a small conv net, keep the joint angles and progress clock, train 20k
steps on 3 seeds. Then the black-image ablation: same model, all-black frames.

*What we hope to learn:* get the whole pixel pipeline running end to end. The
ablation tells me if the model uses the pixels or just rides the clock.

**Outcome.** 0/15, 0/15 and 3/15. With black images it got 0/15 too, with 303
bad frames instead of 259. Not much worse:

- The joint angles and the clock almost predict the next chunk on their own,
  because the cube is in the same place every time here. Pixels barely matter
  when the answer never changes. Like a student acing a test by memorizing the
  order of the answers.

Next levers: the gripper clip, more training, and random cube positions so the
image actually has to matter.

**Takeaway:** always run the blank-input ablation. It's the cheapest way to see
whether your model uses the input you think it does.
