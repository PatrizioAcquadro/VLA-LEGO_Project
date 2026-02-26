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

# When you have a robot MJCF (Phase 1.1+):
# vla-viewer sim/assets/h1_scene.xml --show-contacts --show-joints --camera overhead_cam
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
   - [ ] Box moves when you apply force
   - [ ] Box spins when you apply torque
   - [ ] Box returns to rest after perturbation (no energy explosion)

### Step 5 — Inspect Simulation Info
- Click **Info** in the Option section — shows timestep, solver stats, FPS
- Click **Profiler** — shows timing breakdown of simulation steps
- Click **Sensor** — shows sensor readings (empty for test scene, useful with robots)
- [ ] Info panel displays without errors

### Step 6 — Stability Test
- Let the simulation run for 2–5 minutes
- [ ] No crash, no freeze, no visual glitches
- [ ] FPS stays stable (check bottom-left or Info panel)
- Close the window to exit

## Visual Inspection Checklist (Robot Scenes — Phase 1.1+)

Use this checklist when you load actual robot MJCF files.

### 1. Gravity and Ground Plane
- [ ] Unpause (Space) and confirm the robot falls/settles under gravity
- [ ] Robot does not fall through the floor
- [ ] No body parts start embedded in the ground or in each other

### 2. Joint Axes Directions
- Open **Visualization** panel > enable **Joint**, or launch with `--show-joints`
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
