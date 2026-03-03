# Viewer Debug Checklist (Phase 0.2.2)

Quick reference for visually inspecting a MuJoCo scene using the interactive viewer.

## Launch Commands

```bash
# Basic interactive viewer (blocking — closes on window close)
vla-viewer sim/assets/scenes/test_scene.xml

# With debug overlays (contacts + joints)
vla-viewer sim/assets/scenes/test_scene.xml --show-contacts --show-joints

# Passive mode (steps simulation in a loop, optional timeout)
vla-viewer sim/assets/scenes/test_scene.xml --passive --duration 30

# For robot MJCF (Phase 1.1+):
vla-viewer sim/assets/h1_scene.xml --show-contacts --show-joints --camera overhead_cam
```

## Automated Preflight Checks

The viewer runs these automatically before opening the window:

| Check | What it verifies |
|-------|-----------------|
| **gravity** | Gravity vector is nonzero and points -Z |
| **ground_plane** | At least one plane geom exists in the scene |
| **geom_summary** | Breakdown of collision vs visual-only geoms |

Output example:
```
--- Preflight Checks ---
  [OK] gravity: gravity = [0.00, 0.00, -9.81]
  [OK] ground_plane: 1 plane geom(s) found
  [OK] geom_summary: 3 total geoms: 2 collision, 1 visual-only
------------------------
```

## Walkthrough: test_scene.xml (Current Test Scene)

The test scene contains a ground plane and a falling box. Use this to verify
the viewer works and to learn the UI before loading robot assets.

### Step 1 — Verify Gravity and Ground Contact
1. Launch: `vla-viewer sim/assets/scenes/test_scene.xml`
2. The simulation starts paused. Click **Run** (or press **Space**) to unpause.
3. Watch the box fall. Confirm:
   - [ ] Box falls **downward** (gravity is correct)
   - [ ] Box **lands on the ground** and does not fall through
   - [ ] Box comes to rest (no bouncing forever or explosion)
4. Click **Reset** (or press **Backspace**) to reset the scene to its initial state.
5. Click **Run** again — confirm the same behavior repeats (determinism).

### Step 2 — Navigate the 3D View
Practice camera controls so you can inspect scenes from any angle:
- **Right-click + drag** — Orbit/rotate the camera around the scene
- **Middle-click + drag** (or Shift + right-click) — Pan the camera
- **Scroll wheel** — Zoom in/out
- Try to look at the box from the top, side, and underneath the ground plane
- [ ] Camera controls are responsive and stable

### Step 3 — Explore the Left Panel
The left panel has collapsible sections. Click through each one:

**Simulation section** (already visible):
- **Pause / Run** — toggle simulation stepping
- **Reset** — return to initial state
- **Reload** — re-read the MJCF file from disk (useful after editing XML)

**Rendering section** (click to expand):
- Toggle **Wireframe** — shows mesh edges, useful to see collision geometry
- Toggle **Shadows**, **Reflections**, **Fog** — rendering quality options
- [ ] Toggling wireframe on/off works

**Visualization section** (click to expand):
- **Contact point** — shows magenta dots where bodies touch
- **Contact force** — shows arrows at contact points indicating force magnitude
- **Joint** — shows RGB axes at joint locations
- **Center of mass** — shows body CoM locations
- [ ] Toggle "Contact point" ON, run sim, confirm dots appear where box meets ground
- [ ] Toggle "Contact force" ON, confirm force arrows appear at contact
- [ ] Toggle "Joint" ON, confirm an axis appears on the box's freejoint
- Note: If no Joint ON/OFF toggle is exposed in your viewer build, relaunch with `--passive --show-joints`

**Group enable section** (click to expand):
- Groups 0–5 toggle visibility of different geom groups
- Useful when robot has separate visual and collision meshes
- [ ] Toggling groups on/off changes what is visible

### Step 4 — Apply Perturbation Forces
You can push objects interactively to test dynamics:
1. Make sure the simulation is **running** (click Run)
2. **Double-click** on the box to select it
3. **Ctrl + right-click + drag** on the box — applies a **force** (you will see an arrow)
4. **Ctrl + left-click + drag** on the box — applies a **torque** (rotational force)
5. Confirm:
   - [X] Box moves when you apply force
   - [X] Box spins when you apply torque
   - [X] Box returns to rest after perturbation (no energy explosion)

### Step 5 — Inspect Simulation Info
- Click **Info** in the Option section — shows timestep, solver stats, FPS
- Click **Profiler** — shows timing breakdown of simulation steps
- Click **Sensor** — shows sensor readings (empty for test scene, useful with robots)
- [X] Info panel displays without errors

### Step 6 — Stability Test
- Let the simulation run for 2–5 minutes
- [X] No crash, no freeze, no visual glitches
- [X] FPS stays stable (check bottom-left or Info panel)
- Close the window to exit

## Visual Inspection Checklist (Robot Scenes — Phase 1.1+)

Use this checklist when you load actual robot MJCF files.

### 1. Gravity and Ground Plane
- [ ] Unpause (Space) and confirm the robot falls/settles under gravity
- [ ] Robot does not fall through the floor
- [ ] No body parts start embedded in the ground or in each other

### 2. Joint Axes Directions
- Open **Visualization** panel > enable **Joint**
- If no Joint ON/OFF toggle is exposed, launch with `--passive --show-joints`
- [ ] RGB axes visible at each joint (R=X, G=Y, B=Z)
- [ ] Axes align with expected joint rotation/translation directions
- [ ] No joints located at unexpected positions

### 3. Collision Geoms vs Visual Meshes
- Open **Rendering** panel > toggle **Wireframe** to see collision shapes
- Toggle groups in **Group enable** to show/hide visual vs collision geoms
- [ ] Collision geoms roughly match the shape of visual meshes
- [ ] No collision geoms extend far beyond visual geometry
- [ ] No missing collision geoms on important contact surfaces (hands, feet, table)

### 4. Contact Points and Forces
- Open **Visualization** panel > enable **Contact point** and **Contact force**
- Or launch with `--show-contacts`
- [ ] Contact points appear where expected (feet on ground, fingers on objects)
- [ ] Contact force arrows have reasonable magnitude (no explosions)
- [ ] No spurious contacts at rest pose (penetration at initialization)

### 5. Camera Placement (if cameras are defined in the MJCF)
- Use `--camera <name>` to start from a named camera
- [ ] Camera FOV covers the workspace
- [ ] No important geometry is clipped or occluded
- [ ] Camera is at an appropriate distance for the task

## Keyboard / Mouse Reference

### Mouse Controls
| Action | Effect |
|--------|--------|
| **Right-click + drag** | Rotate / orbit view |
| **Middle-click + drag** | Pan view |
| **Scroll wheel** | Zoom in / out |
| **Double-click body** | Select body (highlights it) |
| **Ctrl + right-click + drag** | Apply perturbation force |
| **Ctrl + left-click + drag** | Apply perturbation torque |

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| **Space** | Pause / unpause simulation |
| **Backspace** | Reset to initial state |
| **Tab** | Toggle left UI panel |
| **Ctrl+A** | Toggle transparency |
| **F1–F5** | Toggle rendering groups (visual/collision geoms) |
| **Escape** or close window | Exit viewer |

## Offscreen Rendering Verification (Phase 0.2.3)

Use this when you need to verify headless offscreen rendering output.

### Run the Validation Script
```bash
python scripts/validate_offscreen.py
```

This produces artifacts in `logs/`:
- `logs/offscreen_smoke.mp4` — 5-second video of the test scene
- `logs/offscreen_frames/` — 5 sample PNG frames at evenly spaced timesteps
- `logs/offscreen_meta.json` — environment and render metadata

### Verify the Video
1. Open `logs/offscreen_smoke.mp4` in any video player (VLC, mpv, etc.)
2. Confirm:
   - [ ] Video is ~5 seconds long and plays smoothly
   - [ ] You can see the ground plane and box
   - [ ] The box falls and hits the ground (not frozen / not black frames)
   - [ ] Colors look correct (gray floor, blue box)

### Verify Sample Frames
1. Open the PNG files in `logs/offscreen_frames/`
2. Confirm:
   - [ ] First frame (`frame_000000_rgb.png`) shows the box at its starting height
   - [ ] Later frames show the box lower / on the ground
   - [ ] Images are not all black or all white
   - [ ] Resolution matches config (default 640x480)

### Verify Depth Output (if enabled)
Run with depth enabled (modify the script or use the API):
```python
from sim.offscreen import RenderConfig, render_trajectory, save_sample_frames
from sim.mujoco_env import load_model
import mujoco

model = load_model("sim/assets/scenes/test_scene.xml")
data = mujoco.MjData(model)
config = RenderConfig(camera_name="overhead", render_depth=True)
frames = render_trajectory(model, data, n_steps=100, config=config, render_every=50)
save_sample_frames(frames, "logs/depth_check/", max_samples=2)
```
- [ ] `frame_*_depth.png` files exist alongside RGB frames
- [ ] Depth images show variation (closer objects are brighter/darker depending on normalization)

## Troubleshooting

### Viewer does not open
- Ensure you are on the lab PC with a physical display (not SSH without X forwarding)
- Check `echo $DISPLAY` returns a value (e.g., `:0` or `:1`)
- Install GLFW if missing: `sudo apt-get install libglfw3 libglfw3-dev`

### Viewer opens but crashes immediately
- Check NVIDIA driver: `nvidia-smi`
- Verify OpenGL works: `glxinfo | grep "OpenGL version"`
- Try with EGL backend: `MUJOCO_GL=egl vla-viewer scene.xml`

### Black screen or no rendering
- Ensure the scene has a light source defined in the MJCF
- Check that visual geoms are not all disabled (toggle rendering groups with F1–F5)

### Perturbation forces do not work
- Make sure the simulation is **running** (not paused)
- Double-click the body first to select it, then Ctrl + drag

### Preflight check shows WARN
- **gravity WARN**: Check `<option gravity="..."/>` in your MJCF
- **ground_plane WARN**: Add a `<geom type="plane" .../>` to the worldbody
- **geom_summary**: Informational only — review collision setup if counts seem wrong

---

## Walkthrough: Alex Upper Body (Phase 1.1.1)

### Launch Command
```bash
vla-viewer sim/assets/scenes/alex_upper_body.xml --show-contacts --show-joints
```

### Step 1 — Verify Fixed Base and Gravity
1. Press **Space** to unpause
2. Confirm:
   - [X] Robot does NOT fall (fixed base, no freejoint)
   - [X] Arms hang naturally under gravity (settle to a rest pose)
   - [X] No body parts embedded in the floor or each other

### Step 2 — Verify Joint Axes
1. Ensure joint axes are enabled:
   - Toggle **Joint** in the Visualization panel, or
   - Relaunch with `--passive --show-joints` if the toggle is not exposed
2. Confirm:
   - [X] SPINE_Z axis is vertical (blue Z arrow at torso)
   - [ ] Shoulder joints have axes aligned with expected Y/X/Z rotations
   - [ ] Elbow joint axis is horizontal (Y axis)
   - [X] Wrist and gripper joints are at the distal arm end
   - [X] No joints appear on the head (neck joints were removed)

### Step 3 — Verify Collision Geometry
1. Toggle **Wireframe** in Rendering panel, or press **F3** to show group 3
2. Confirm:
   - [ ] Collision capsules/boxes visible on torso, shoulders, biceps, forearms, wrists, grippers
   - [ ] Collision geoms roughly envelope the visual meshes (not oversized, not missing)
   - [ ] Floor plane has collision

### Step 4 — Verify EE Sites
1. Enable **Site** visualization (group 4) in the viewer
2. Confirm:
   - [ ] Red sphere at left gripper tip (left_ee_site)
   - [ ] Blue sphere at right gripper tip (right_ee_site)

### Step 5 — Perturbation Test
1. Double-click the left forearm, Ctrl+right-drag to apply force
2. Confirm:
   - [ ] Arm moves and returns toward rest (PD actuators active)
   - [ ] No explosion or instability
   - [ ] Repeat for right arm

### Step 6 — Camera Views
1. `vla-viewer sim/assets/scenes/alex_upper_body.xml --camera overhead`
2. `vla-viewer sim/assets/scenes/alex_upper_body.xml --camera third_person`
3. Confirm:
   - [ ] `overhead` shows the full robot from above, workspace visible
   - [ ] `third_person` shows the robot from an angle suitable for observation

### What "correct" looks like
- Robot stands stationary with arms at sides or slightly forward
- No jitter, no drift, no NaN warnings in console
- 15 joint axes visible (1 spine + 7 left + 7 right)
- Collision geoms form a reasonable bounding envelope

### What "wrong" looks like
- Robot collapses, explodes, or drifts
- Arms clip through torso
- Joints at unexpected locations or with wrong axis directions
- Collision geoms wildly different from visual mesh sizes

---

## Walkthrough: Alex Dynamics Validation (Phase 1.1.2)

### Launch Command
```bash
vla-viewer sim/assets/scenes/alex_upper_body.xml --show-contacts --show-joints
```

### Step 1 — Verify PD Hold at Home Pose
1. Press **Space** to unpause
2. Wait 5 seconds for the robot to settle
3. Confirm:
   - [ ] Arms settle to a stable rest position (no oscillation)
   - [ ] No visible jitter in any joint
   - [ ] Velocity readings near zero after settling

### Step 2 — Perturbation Recovery
1. Double-click left forearm, Ctrl+right-drag to apply moderate force
2. Confirm:
   - [ ] Arm deflects and returns smoothly to rest
   - [ ] No overshoot oscillation (damping is adequate)
   - [ ] Recovery takes ~0.5-2s (not too sluggish, not too fast)
3. Repeat with right arm

### Step 3 — Verify Contact Solver Quality
1. Enable **Contact point** and **Contact force** visualization
2. Let robot settle under gravity
3. Confirm:
   - [ ] No spurious contact points flickering on/off
   - [ ] Contact forces are smooth, not oscillating

### Step 4 — Test Rest Keyframe
1. In the viewer, load the `rest` keyframe (shoulders abducted, elbows bent)
2. Confirm:
   - [ ] Robot reaches the rest pose without collision
   - [ ] Arms stay stable in the bent-elbow configuration
   - [ ] No self-collision warnings

### What "correct" looks like
- Robot holds position with zero visible drift
- Perturbations produce smooth, damped recovery
- No high-frequency jitter anywhere

### What "wrong" looks like
- Arms oscillate around the target position (damping too low)
- Arms barely move when perturbed (damping too high)
- Contacts flicker rapidly (solver iterations too low)
- Joint positions drift slowly over time (friction too low)

---

## Walkthrough: EZGripper Integration (Phase 1.1.3)

### Launch Command
```bash
vla-viewer sim/assets/scenes/alex_upper_body.xml --show-contacts --show-joints
```

### Step 1 — Verify EZGripper Geometry
1. Press **Space** to unpause, let robot settle (2-3 seconds)
2. Zoom in on left hand (scroll + right-click drag)
3. Confirm:
   - [ ] EZGripper palm mesh visible at end of each arm (replaces nub)
   - [ ] Two finger pairs visible on each hand (L1 + L2 per finger)
   - [ ] Fingers are in closed position at home (joint = 0)
   - [ ] No visual gaps between palm and wrist roll link
   - [ ] Palm orientation looks correct relative to forearm

### Step 2 — Verify Finger Collision Geometry
1. Press **F3** to toggle collision geom visibility (group 3)
2. Confirm:
   - [ ] Box collision primitives visible on each finger segment (green-ish)
   - [ ] Palm has a box collision geom
   - [ ] Collision boxes roughly match finger pad dimensions (~30mm × 50mm)
   - [ ] No collision geoms floating detached from visual meshes

### Step 3 — Verify Tool Frame Sites
1. Enable **Site** visualization (group 4) in the viewer
2. Confirm:
   - [ ] Red sphere at left gripper (left_ee_site)
   - [ ] Blue sphere at right gripper (right_ee_site)
   - [ ] Yellow spheres at EZGripper palm centers (left/right_tool_frame)

### Step 4 — Test Open/Close via Keyframes
1. Load the `open_grippers` keyframe in the viewer
2. Confirm:
   - [ ] All 4 finger joints on left hand open together (coupling works)
   - [ ] All 4 finger joints on right hand open together
   - [ ] Motion is smooth, no jitter
3. Load the `home` keyframe — fingers should close

### Step 5 — Contact Test
1. Enable **Contact point** and **Contact force** visualization
2. Close grippers (home keyframe, actuator ctrl = 0)
3. Confirm:
   - [ ] Finger pads of opposing fingers create contact when closed
   - [ ] Contact forces are reasonable (not exploding)
   - [ ] No spurious contacts between palm and fingers

### Step 6 — Grasp Test Scene
```bash
vla-viewer sim/assets/scenes/alex_grasp_test.xml --show-contacts
```
1. Load `pregrasp` keyframe (arms forward, grippers open)
2. Note: pre-grasp keyframe is approximate; may need manual tuning
3. Confirm:
   - [ ] Table and red cube visible in front of the robot
   - [ ] Cube sits stably on the table
   - [ ] Arms positioned roughly near the cube (adjust keyframe if needed)

### What "correct" looks like
- EZGripper palm attached at correct orientation to wrist
- Two-finger parallel gripper opening/closing smoothly
- All 4 joints per hand move in unison (equality coupling)
- Finger pads make clean contact with grasped objects

### What "wrong" looks like
- Palm rotated 90° or flipped (adapter euler transform needs adjustment)
- Fingers not coupled (one moves, others stay static)
- Explosive contact forces when fingers touch each other or objects
- Collision geoms far larger/smaller than visual fingers

---

## Walkthrough: Kinematics Validation (Phase 1.1.4)

Visual verification of arm symmetry, joint axes, and workspace reachability.

### Launch Command
```bash
vla-viewer sim/assets/scenes/alex_upper_body.xml --show-joints
```

### Step 1 — Verify Left/Right Arm Symmetry
1. Press **Space** to unpause, let robot settle (2-3 seconds)
2. Orbit the view to look at the robot from the front
3. Confirm:
   - [ ] Left and right arms are visually symmetric (mirror image about the midline)
   - [ ] EE sites (red/blue spheres, group 4) are at the same height and symmetric Y positions
   - [ ] No arm is significantly longer or shorter than the other

### Step 2 — Verify Joint Axis Directions
1. Enable **Joint** visualization (Visualization panel or `--show-joints`)
2. Zoom in on the left shoulder area
3. Confirm:
   - [ ] shoulder_y axis is green (Y-axis) — rotation abducts/adducts arm
   - [ ] shoulder_x axis is red (X-axis) — rotation lifts arm forward/backward
   - [ ] shoulder_z axis is blue (Z-axis) — rotation twists the upper arm
4. Check elbow:
   - [ ] elbow_y axis is green (Y-axis) — flexion/extension
5. Check wrist:
   - [ ] wrist_z axis is blue (Z-axis) — forearm rotation
   - [ ] wrist_x axis is red (X-axis) — wrist flexion
6. Check gripper:
   - [ ] gripper_z axis is blue (Z-axis) — gripper rotation

### Step 3 — Verify Symmetric Pose Behavior
1. Double-click left forearm, Ctrl+right-drag to perturb it
2. Repeat the same perturbation on the right forearm
3. Confirm:
   - [ ] Both arms respond similarly to the same direction of force
   - [ ] Recovery motion is a mirror image (left moves +Y, right moves -Y)
   - [ ] No unexpected asymmetry in motion range or speed

### Step 4 — Verify Workspace Reachability
1. Load the `rest` keyframe (shoulders abducted, elbows bent)
2. Confirm:
   - [ ] Both hands are positioned in front of the robot's torso
   - [ ] Hands are roughly at table height (~0.8–1.0m from ground)
   - [ ] Arms can visually reach the table area in the grasp test scene

### What "correct" looks like
- Perfect visual symmetry between left and right arms
- Joint axes follow RGB=XYZ convention at expected locations
- Perturbations produce mirror-image responses
- Both arms can reach the workspace in front of the robot

### What "wrong" looks like
- One arm hangs differently from the other at home pose (axis flip)
- Joint axis arrows point in unexpected directions
- Perturbation on left arm causes movement in a completely different direction than right arm
- Arms cannot reach forward past the torso (workspace issue)

---

## Walkthrough: Action Space Verification (Phase 1.1.5)

Visual verification that the 17-D action space produces stable, bounded motion.

### Launch Command
```bash
vla-viewer sim/assets/scenes/alex_upper_body.xml --show-contacts --show-joints
```

### Step 1 — Run Validation Script First
```bash
python scripts/validate_action_space.py
```
Check terminal output for PASS/FAIL on random actions and action chunks.
Open `logs/action_space_validation/random_actions.mp4` to see the robot moving.
- [ ] Video shows smooth, bounded motion — no explosions or NaN freezes
- [ ] Arms move in varied directions (random actions are applied)
- [ ] Grippers open/close visibly during the video

### Step 2 — Interactive Stability Check
1. Press **Space** to unpause, let robot settle (2-3 seconds)
2. Apply perturbation force to left forearm (Ctrl+right-drag)
3. Confirm:
   - [ ] Arm deflects and returns to rest (PD control active)
   - [ ] No oscillation or jitter after perturbation
4. Apply perturbation to right forearm — same behavior expected

### Step 3 — Verify Contact Stability Under Motion
1. Enable **Contact point** and **Contact force** visualization
2. Let robot settle, then apply perturbation to move arms toward each other
3. Confirm:
   - [ ] If arms touch, contact forces are bounded (no explosive arrows)
   - [ ] Contact points are stable (not flickering rapidly)
   - [ ] No deep interpenetration visible

### Step 4 — Verify Gripper During Motion
1. Load `open_grippers` keyframe, then `home` keyframe
2. While robot arms are settling, confirm:
   - [ ] Grippers open/close cleanly during arm motion
   - [ ] Finger coupling still works (all 4 joints per hand move together)
   - [ ] No instability when grippers change state

### What "correct" looks like
- Robot executes smooth, bounded joint motions
- No jitter, drift, or NaN warnings even after many actions
- Grippers and arms operate independently without interference
- Contact forces during collisions are bounded and stable

### What "wrong" looks like
- Arms oscillate rapidly (excessive delta_q_max or poor damping)
- Robot freezes (NaN in qpos — action pipeline bug)
- Arms hit joint limits and jitter there (rate limiting not working)
- Gripper commands cause arm motion to destabilize

---

## Walkthrough: Multi-View Camera Verification (Phase 1.1.7)

### Launch Command — Robot Camera (Ego-Centric)
```
vla-viewer sim/assets/scenes/alex_grasp_test.xml --camera robot_cam
```

### Step 1 — Verify robot_cam Perspective
1. Launch with `--camera robot_cam`
2. The view renders the robot's head-mounted ego-centric perspective
3. Confirm:
   - [ ] View is looking forward/downward at the workspace area
   - [ ] Both arms are partially visible at the lower edges of the frame
   - [ ] The table surface and graspable objects are visible
   - [ ] Ground plane is visible in the distance
4. Enable simulation and apply perturbation to the torso:
   - [ ] Camera view rotates/shifts with the torso (spine_z rotation)
   - [ ] View remains stable (no jitter or sudden jumps)

### Step 2 — Verify third_person Camera (External)
1. Restart with:
   ```
   vla-viewer sim/assets/scenes/alex_grasp_test.xml --camera third_person
   ```
2. Confirm:
   - [ ] Full robot is visible from an elevated side angle
   - [ ] Both arms, torso, and workspace area are in frame
   - [ ] The camera does NOT move when the spine rotates
   - [ ] Good overview framing for debugging and presentation

### Step 3 — Compare Offscreen Renders
1. Run the validation script:
   ```
   python scripts/validate_cameras.py
   ```
2. Check `logs/camera_validation/robot_cam_sample.png`:
   - [ ] Matches what you saw in the viewer for the robot_cam perspective
   - [ ] Non-black, reasonable exposure
3. Check `logs/camera_validation/third_person_sample.png`:
   - [ ] Matches what you saw in the viewer for the third_person perspective
   - [ ] Full robot visible
4. Check `logs/camera_validation/dual_view_video.mp4`:
   - [ ] Both views animate simultaneously (side-by-side)
   - [ ] robot_cam view changes as robot moves; third_person stays stable
   - [ ] No frame drift or synchronization issues
5. Check `logs/camera_validation/camera_metadata.json`:
   - [ ] Contains fovy, position, and orientation for both cameras
   - [ ] robot_cam fovy = 60, third_person fovy = 50

### What "correct" looks like
- robot_cam shows a natural ego perspective of the table/workspace
- third_person shows the full robot from an elevated side angle
- Both views render non-black frames with proper lighting
- robot_cam position changes when spine rotates; third_person stays fixed
- Dual-view video is synchronized (both views show the same simulation state)

### What "wrong" looks like
- Black frames: renderer issue or camera pointing at empty space
- robot_cam not moving with spine: camera not body-attached (MJCF error)
- Both cameras pointing at sky/floor only: position/orientation error in MJCF
- Flickering or frame misalignment in dual-view video: synchronization bug
