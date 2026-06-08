# Sensor test scenarios

Three realistic occupancy scripts to act out in front of the smartroom node.
Each is a normal way a person uses a room — no posing or touching the sensors —
so a recording captures lifelike multimodal data. Each maps to a `scenario_id`
you can record in `metadata.json`.

## How to run one

1. Start the recording: `python capture.py` (records 30s by default).
2. As soon as it prints `Recording...`, **clap once** — this is the `t=0` sync
   marker. It shows up as a spike in the mic stream and a frame in the video, so
   you can line all the streams up afterward (`sync_method: clap_at_t0`).
3. Perform the beats below against a clock/phone timer.
4. Check the stream files to confirm the activity was captured.

| Sensor | Stream | What it picks up during occupancy |
|---|---|---|
| Camera | `camera_main.mp4` | presence, movement, posture |
| OPS243-A radar | `radar_ops243.csv` | motion **toward/away** from the node (radial speed) |
| MCP3008 mic | `mcp3008_mic.csv` | speech, footsteps, ambient noise |
| BME680 / TCS / ADXL / MLX | `custom_board_i2c.csv` | slow context: temp/humidity, room light, ambient — mostly steady when no one touches the board |

> Radar responds to **radial** motion — walking *toward/away* from the node
> registers strongly; walking *across* it barely shows. Aim movement at the node.

---

## Scenario 1 — `walkthrough_v1` (someone passes through, 30s)

A person crosses the room and leaves — transient occupancy, room ends empty.

| Time | Action |
|---|---|
| 0:00 | Off to the side, clap once (sync) |
| 0:03 | Walk into the room toward the node |
| 0:09 | Slow as you pass the node, glance at it |
| 0:14 | Continue across to the far side |
| 0:20 | Walk back out the way you came |
| 0:27 | Leave the room |
| 0:30 | End (empty) |

---

## Scenario 2 — `desk_work_v1` (sedentary occupancy, 30s)

One person sits and works the whole time — sustained presence, little movement.
The hard case for motion sensors.

| Time | Action |
|---|---|
| 0:00 | Clap once (sync) |
| 0:02 | Walk in and sit down within view |
| 0:06 | Settle, type / read / look at a phone |
| 0:14 | Say a few sentences as if on a call |
| 0:22 | Go back to quiet work, occasional small shifts |
| 0:28 | Still and quiet |
| 0:30 | End (still seated) |

---

## Scenario 3 — `active_occupancy_v1` (moving around, 30s)

One person actively using the space — pacing, fetching things, multitasking.

| Time | Action |
|---|---|
| 0:00 | Clap once (sync) |
| 0:03 | Walk in and move toward the node |
| 0:08 | Pace around, pick something up, set it down |
| 0:14 | Talk out loud while moving |
| 0:19 | Step away from the node, then come back |
| 0:25 | Stop and stand near the node |
| 0:30 | End (still present) |

---

## Notes

- Vary which scenario you record so the dataset spans transient, sedentary, and
  active occupancy rather than one mode.
- The i2c context sensors stay fairly flat during normal occupancy — that's
  expected; they capture room conditions, not the person.
