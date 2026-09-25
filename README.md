# Cat Bowl Monitor

A Raspberry Pi with a camera watches your cats' water bowl from above and logs which cat visits and for how long. It tells the cats apart by their coats: one is white with black patches, the other white with orange patches. A small image-recognition model, trained on photos of your own cats, does the
recognizing.

This guide assumes no prior experience with Linux, Docker, or machine learning. Follow it top to bottom the first time.

---

## Contents

1. [How the whole system works](#1-how-the-whole-system-works)
2. [What you need](#2-what-you-need)
3. [Set up your Windows computer](#3-set-up-your-windows-computer)
4. [Set up the Raspberry Pi and connect it to Wi-Fi](#4-set-up-the-raspberry-pi-and-connect-it-to-wi-fi)
5. [Connect to the Pi from your computer](#5-connect-to-the-pi-from-your-computer)
6. [Install the software on the Pi](#6-install-the-software-on-the-pi)
7. [Aim the camera and set the crop](#7-aim-the-camera-and-set-the-crop)
8. [Collect training photos](#8-collect-training-photos)
9. [Label the photos](#9-label-the-photos)
10. [Train and test the model](#10-train-and-test-the-model)
11. [Deploy the model to the Pi](#11-deploy-the-model-to-the-pi)
12. [Run it automatically at startup](#12-run-it-automatically-at-startup)
13. [Everyday use](#13-everyday-use)
14. [Improving the model over time](#14-improving-the-model-over-time)
15. [Changing Wi-Fi networks](#15-changing-wi-fi-networks)
16. [Troubleshooting](#16-troubleshooting)
17. [Command cheat sheet](#17-command-cheat-sheet)

---

## 1. How the whole system works

There are two machines:

- **Your desktop computer** is where you label photos, train the model, and
  edit settings. All of that happens inside a "container": a sealed,
  pre-configured Linux environment that Docker runs, so every tool is the
  exact right version.
- **The Raspberry Pi** lives in the Pelican case, takes
  photos, and, once trained, runs the model to watch the bowl.

The workflow, in order:

```
 Pi: collect photos ──► Computer: label photos ──► Computer: train + test model
                                                               │
 Pi: watch bowl, log visits ◄── copy model to Pi ◄─────────────┘
```

### Four different terminals: know which one you're in

This guide uses four terminal windows, and
every command block starts with a comment saying which one:

| Label in this guide | What it is | How to open it | What the prompt looks like |
|---|---|---|---|
| `# Where: PowerShell` | Windows' own terminal | Start menu → type "PowerShell" | `PS C:\Users\you>` |
| `# Where: WSL` | Linux (Ubuntu) running on Windows | Start menu → "Ubuntu" | `you@PC:~$` |
| `# Where: Container` | The terminal inside VS Code once the project is open in the container | VS Code → Terminal → New Terminal | `dev@abc123:/workspace$` |
| `# Where: Pi` | A remote connection into the Raspberry Pi | From WSL, run `ssh $PI` (set up in section 5) | `yourname@catcam:~ $` |

Anything in `CAPITALS_LIKE_THIS` is a placeholder. Replace it with your own
value.

---

## 2. What you need

### Hardware

- Raspberry Pi 4 or 5
- microSD card, 32 GB or larger
- A Raspberry Pi ribbon-cable camera (e.g. Camera Module 3), with the correct cable for your Pi.
  **Pi 5 owners:** the Pi 5 has a smaller camera connector and needs the "22-pin to 15-pin" camera cable. Most cameras ship with the Pi 4 cable.
- Power:
  - Pi 5: official 27 W USB-C supply
  - Pi 4: official 15 W supply
  - A battery pack works for short sessions only (about 6–10 hours).
- The Pelican case, with a hole or window for the camera to see through
- Optional: the small screen, plus a USB keyboard for emergencies
- A computer to write the SD card (your Windows computer with a card reader)

### Software (installed in section 3)

- Windows 10 (version 2004 or newer) or Windows 11
- WSL (Windows Subsystem for Linux)
- Docker Desktop
- Visual Studio Code
- Raspberry Pi Imager

---

## 3. Set up your Windows computer

### 3.1 Install WSL (Linux on Windows)

1. Right-click the Start button and choose **Terminal (Admin)** or
   **Windows PowerShell (Admin)**.
2. Run:

```powershell
   # Where: PowerShell (Admin)
   wsl --install
```

3. Restart the computer when it finishes.
4. After the restart, an **Ubuntu** window opens by itself. If it doesn't, open "Ubuntu" from the Start menu. It asks you to create a username and password. These are for Linux only and can be anything. The password doesn't show as you type; that's normal.

Install some basic tools in Ubuntu:

```bash
# Where: WSL
sudo apt update
sudo apt install -y git rsync openssh-client python3-venv python3-pip libgl1
```

### 3.2 Install Docker Desktop

1. Download it from <https://www.docker.com/products/docker-desktop/> and install it with the default options.
2. Open Docker Desktop, then click the **gear icon (Settings)**.
3. **General**: make sure "Use the WSL 2 based engine" is checked.
4. **Resources → WSL integration**: turn on the switch for **Ubuntu**. Click
   **Apply & restart**.

Check that it works:

```bash
# Where: WSL
docker run --rm hello-world
```

You should see "Hello from Docker!".

### 3.3 Install VS Code and its extensions

1. Install VS Code from <https://code.visualstudio.com/>.
2. Open it and click the Extensions icon (four squares) on the left.
3. Search for and install **WSL** (by Microsoft) and **Dev Containers** (by Microsoft).

### 3.4 Tell git who you are (one time)

```bash
# Where: WSL
git config --global user.name "YOUR NAME"
git config --global user.email "YOUR_EMAIL@example.com"
```

### 3.5 Get the project code

Keep the project **inside Linux**, in your Linux home folder, not on `C:`.

```bash
# Where: WSL
mkdir -p ~/code
cd ~/code
git clone REPOSITORY_URL cat-bowl-monitor
cd cat-bowl-monitor
```

Replace `REPOSITORY_URL` with the address of wherever this repo is hosted. If GitHub asks for a password, it wants a personal access token, not your account password. See GitHub's help for "personal access token".

### 3.6 Create the secrets file and the data-storage folder

```bash
# Where: WSL (inside ~/code/cat-bowl-monitor)
cp .env.example .env
mkdir -p ~/dvc-storage
```

`.env` holds passwords or keys if you ever need them. None are needed today, but the container won't start without the file.

`~/dvc-storage` is where old versions of your photo collection are stored. The container needs permission to see it. Open `.devcontainer/devcontainer.json` in any editor and add one line to the `"mounts"` list, so it looks like this:

```jsonc
  "mounts": [
    "source=/tmp/.X11-unix,target=/tmp/.X11-unix,type=bind",
    "source=/mnt/wslg,target=/mnt/wslg,type=bind",
    "source=${localEnv:HOME}/dvc-storage,target=/dvc-storage,type=bind"
  ],
```

Note the comma at the end of the second line.

`~/dvc-storage` is on the same disk as your project. It protects against mistakes, not against disk failure. For a real backup, copy it to an external drive now and then, or use cloud storage (section 9.4).

### 3.7 Open the project in the container

Open VS Code and make sure it is opening in WSL as indicated in the bottom-left corner. A pop-up in the bottom-right says "Folder contains a Dev Container configuration file". Click **Reopen in Container**. If you miss it, press **F1**, type `Reopen in Container`, and press Enter.

The first time, this downloads and builds everything. It can take 10–30 minutes. Later openings take seconds.

When it's done, the green or blue box at the bottom-left of VS Code says **Dev Container: cat-bowl-monitor**. Open a terminal with **Terminal → New Terminal**.

### 3.8 Test the computer setup

```bash
# Where: Container
make format
make verify
```

The last line should report tests **passed** (currently 7 tests). If anything fails, see [Troubleshooting](#16-troubleshooting) before continuing.

### 3.9 Set up version control for data (one time)

```bash
# Where: Container
git add -A
git commit -m "Initial project"
dvc init
dvc remote add -d storage /dvc-storage
git add -A
git commit -m "Set up DVC"
```

DVC ("Data Version Control") keeps track of the photo collection the way git
tracks code, without stuffing gigabytes of photos into git.

## 4. Set up the Raspberry Pi and connect it to Wi-Fi

### 4.1 Write the operating system to the SD card

1. Install **Raspberry Pi Imager** on Windows from
   <https://www.raspberrypi.com/software/>.
2. Put the microSD card in your computer and open Imager.
3. **Choose Device**: your Pi model (Pi 4 or Pi 5).
4. **Choose OS**: **Raspberry Pi OS (64-bit)**, the recommended one with desktop.
5. **Choose Storage**: your SD card. Double-check it's the card, because this erases it.
6. Click **Next**. When asked "Would you like to apply OS customisation settings?", click **Edit Settings** and fill in:
   - **General** tab:
     - **Set hostname**: `catcam`
     - **Set username and password**: pick a username (e.g. your first name,
       lowercase) and a password. **Write both down.** This guide calls the
       username `YOURNAME`.
     - **Configure wireless LAN**: your Wi-Fi name (SSID) and password.
       **Wireless LAN country**: `US`. Wi-Fi names are case-sensitive.
     - **Set locale settings**: your time zone (e.g. `America/New_York`).
   - **Services** tab: check **Enable SSH** and choose **Use password
     authentication**.
7. Click **Save**, then **Yes** to apply the settings, then **Yes** to erase and write. Wait for "Write Successful" and remove the card.

The Pi only joins the Wi-Fi network you enter here, so do this at home. To move the Pi to a different network later, see [section 15](#15-changing-wi-fi-networks).

### 4.2 Connect the camera (Pi unplugged)

**Always unplug the power before connecting or disconnecting the camera.**

1. Find the camera connector: on a Pi 4 it's labeled **CAMERA**; on a Pi 5, use **CAM/DISP 0**.
2. Gently lift the connector's plastic latch.
3. Slide the ribbon in with the metal contacts facing the right way. Follow the photo for your exact Pi model in the official guide: <https://www.raspberrypi.com/documentation/accessories/camera.html>. Backwards is the most common mistake.
4. Push the latch back down. The ribbon should not pull out with a light tug.

### 4.3 First boot

1. Insert the SD card into the Pi.
2. Plug in the power.
3. Wait **3–5 minutes**. The first boot sets itself up and may restart once.

---

## 5. Connect to the Pi from your computer

### 5.1 Find the Pi's address

WSL usually can't find the Pi by its name (`catcam.local`), so get its numeric address from Windows:

```powershell
# Where: PowerShell (normal, not admin)
ping -4 catcam.local
```

You'll see lines like `Reply from 192.168.1.57`. That number is the Pi's IP address. Write it down.

If ping says it can't find the host, try one of these:

- Wait two more minutes and try again.
- Log into your router's admin page and look in the list of connected devices for `catcam`.
- Plug the screen and keyboard into the Pi, log in, open a terminal, and type `hostname -I`.

**Recommended:** in your router's settings, create a "DHCP reservation" (also called an "address reservation" or "static lease") for `catcam`. Then its address never changes. Otherwise it may change after a router restart, and
you'll need to repeat this step.

### 5.2 Save the Pi's address as a shortcut

Replace `YOURNAME` and the IP with yours:

```bash
# Where: WSL
echo 'export PI=YOURNAME@192.168.1.57' >> ~/.bashrc
source ~/.bashrc
echo $PI
```

The last command should print your `YOURNAME@address`. From now on, `$PI` means "the Pi".

### 5.3 Log into the Pi for the first time

```bash
# Where: WSL
ssh $PI
```

It asks "Are you sure you want to continue connecting?". Type `yes`, then enter the Pi password you chose in Imager. The prompt changes to `YOURNAME@catcam:~ $`, which means you're now typing commands **on the Pi**.

To leave the Pi and return to WSL, type `exit`.

### 5.4 Stop typing the password every time (recommended)

```bash
# Where: WSL
ssh-keygen -t ed25519
# Press Enter at every question to accept the defaults.
ssh-copy-id $PI
# Enter the Pi password one last time.
ssh $PI
# It should log in without asking for a password. Type `exit` to leave.
```

---

## 6. Install the software on the Pi

### 6.1 Update the Pi and check the time

```bash
# Where: Pi
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

The connection closes during the reboot. Wait a minute, then run `ssh $PI` again from WSL.

Check that the clock and time zone are right. Visit timestamps depend on it:

```bash
# Where: Pi
timedatectl
```

If the time zone is wrong, set it (example for Virginia):

```bash
# Where: Pi
sudo timedatectl set-timezone America/New_York
```

The Pi gets the correct time from the internet, so keep it on Wi-Fi. A Pi 4 has no clock battery, and a Pi 5's clock battery is optional. Without internet, the time can be wrong after a restart.

### 6.2 Test the camera

```bash
# Where: Pi
rpicam-hello --list-cameras
```

It should list one camera, e.g. `imx708`. If it says **No cameras available**, see [Troubleshooting](#16-troubleshooting).

Take a test photo:

```bash
# Where: Pi
rpicam-still -o ~/test.jpg
exit
```

Copy it to your computer and look at it:

```bash
# Where: WSL
scp $PI:~/test.jpg ~/code/cat-bowl-monitor/
cd ~/code/cat-bowl-monitor
explorer.exe .
```

A Windows folder window opens. Double-click `test.jpg`. You should see a real, in-focus picture.

### 6.3 Install system packages

```bash
# Where: Pi
sudo apt install -y python3-picamera2 python3-opencv python3-numpy tmux
```

The camera library must come from `apt` like this, not from `pip`. Mixing the two breaks it.

### 6.4 Copy the project code to the Pi

Run this from your computer whenever you've changed code or `config.yaml`. It copies code only, never your data, models, or secrets:

```bash
# Where: WSL
cd ~/code/cat-bowl-monitor
rsync -av \
  --exclude '.git' --exclude '.dvc' --exclude '.venv*' --exclude '__pycache__' \
  --exclude 'data/' --exclude 'runs/' --exclude 'models/' \
  --exclude 'delete_me/' --exclude '.env' \
  ./ $PI:~/cat-bowl-monitor/
```

This guide calls it **"sync code to the Pi."**

### 6.5 Create the Python environment on the Pi (one time)

```bash
# Where: Pi
cd ~/cat-bowl-monitor
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-pi.txt
```

The flag `--system-site-packages` lets this environment use the camera library installed with `apt` in 6.3.

Test that everything imports:

```bash
# Where: Pi
python -c "from picamera2 import Picamera2; import cv2, onnxruntime, yaml; print('Pi software OK')"
```

It should print `Pi software OK`.

**Every time you log into the Pi to run project commands, start with:**

```bash
# Where: Pi
cd ~/cat-bowl-monitor
source .venv/bin/activate
```

The prompt then starts with `(.venv)`. The rest of this guide assumes you've done this.

---

## 7. Aim the camera and set the crop

The model only looks at one rectangle of the picture, called the "crop". It should include the bowl **and** the area where a drinking cat's back and shoulders are. The coat patches are often on the back, not the head.

**Decide the crop before labeling.** Changing it later can make old labels wrong.

### 7.1 Mount everything in place

Put the Pelican case and camera in their final position above the bowl, and make sure the camera can't shift. A camera that moves makes the crop point at the wrong place.

### 7.2 Take a check picture

```bash
# Where: Pi
python -m catwatch.check_crop
```

It prints the frame size, e.g. `1280 x 960 pixels`, and saves a picture. Copy it to your computer to look at it:

```bash
# Where: WSL
mkdir -p ~/code/cat-bowl-monitor/data
scp "$PI:~/cat-bowl-monitor/data/crop_check_*.jpg" ~/code/cat-bowl-monitor/data/
explorer.exe ~/code/cat-bowl-monitor/data
```

Pixel positions count from the top-left corner: `x` goes right, `y` goes down.

### 7.3 Set the crop and focus

In VS Code, open `config.yaml` and edit the `bowl_crop` line. For example:

```yaml
bowl_crop: {x: 240, y: 80, width: 800, height: 800}
```

This means: start 240 pixels from the left and 80 from the top, 800 wide and 800 tall.

If your camera has autofocus (Camera Module 3 does), fix the focus so it doesn't hunt when a cat walks in. Measure from the lens to the bowl in metres, then set `lens_position` to **1 ÷ that distance**. For example:

```yaml
lens_position: 2.0     # bowl is 0.5 m away → 1 ÷ 0.5 = 2.0
```

Then sync code to the Pi (command in 6.4) and run `check_crop` again. Look at both pictures:

- `crop_check_full.jpg` shows the whole view with a yellow rectangle.
- `crop_check_crop.jpg` shows just what the model will see.

Repeat until the crop is right and the image is sharp.

When you're happy, save the settings in git:

```bash
# Where: Container
git add config.yaml
git commit -m "Set bowl crop and focus"
```

---

## 8. Collect training photos

The collector saves a photo when something moves, at most one every 2 seconds. It also saves one photo every 10 minutes no matter what, so the model sees the empty bowl in every kind of lighting.

### 8.1 Start collecting

Use `tmux`, a program that keeps things running after you disconnect:

```bash
# Where: Pi
tmux new -s collect
cd ~/cat-bowl-monitor
source .venv/bin/activate
python -m catwatch.collect
```

It prints `Collecting images. Press Ctrl+C to stop.`

Now **detach**: press **Ctrl+B**, let go, then press **D**. The collector keeps running, and you can close the window or type `exit`.

To check on it later:

```bash
# Where: Pi
tmux attach -t collect
```

Detach again with Ctrl+B, then D.

### 8.2 How long to collect

Leave it for **about a week**, or until you have photos from many different times of day. As a rough target, aim for 200–300 good photos of **each** cat, spread over many different hours.

If the room is dark at night and the cats drink then, night photos are important. Colors fade in the dark, so consider a small night light near the bowl.

### 8.3 Check health while collecting (optional)

```bash
# Where: Pi
vcgencmd measure_temp       # under ~75'C is comfortable
vcgencmd get_throttled      # "throttled=0x0" means no power or heat problems
df -h ~                     # "Avail" shows free space on the SD card
```

### 8.4 Stop collecting

```bash
# Where: Pi
tmux attach -t collect
# press Ctrl+C
exit
```

The Pi can only run **one** camera program at a time: collect, check_crop, or monitor. Stop one before starting another.

### 8.5 Copy the photos to your computer

```bash
# Where: WSL
mkdir -p ~/code/cat-bowl-monitor/data
rsync -av $PI:~/cat-bowl-monitor/data/raw/ ~/code/cat-bowl-monitor/data/raw/
```

This guide calls it **"pull photos from the Pi."** It's safe to run repeatedly; it only copies new files.

Open a few photos before labeling to make sure they're sharp and framed correctly:

```bash
# Where: WSL
explorer.exe ~/code/cat-bowl-monitor/data/raw
```

---

## 9. Label the photos

Labeling means telling the computer what's in each photo. It's the most important step: the model can only be as good as its labels.

### 9.1 Start the labeling tool

```bash
# Where: Container
python -m catwatch.label --labeled-by YOURNAME
```

A window opens with a photo and a yellow rectangle (the crop). **Click on the window** so it receives your key presses, then press one key per photo:

| Key | Meaning |
|---|---|
| `0` | Empty: no cat in the rectangle |
| `1` | The cat with **black** patches |
| `2` | The cat with **orange** patches |
| `3` | Both cats |
| `x` | Unusable: can't tell (blurry, only a tail, a person's hand, etc.) |
| `z` | Go back one photo, to fix a mistake |
| `q` | Quit. Progress is saved; next time it continues where you left off. |

If no window appears, see "The labeling window never appears" in [Troubleshooting](#16-troubleshooting).

### 9.2 Labeling rules (follow these every time)

1. **Only judge what's inside the yellow rectangle.** That's all the model sees.
2. **Label a cat** only if you could tell which cat it is from inside the rectangle alone.
3. **When in doubt, press `x`.** A skipped photo costs nothing; a wrong label teaches the model the wrong thing.
4. **Only label what you see.** Don't label from memory ("that was probably Orion at 3 pm").
5. Using `z` and re-labeling is fine. The newest label wins, and the old one stays in the history.

### 9.3 Save this labeling round as a version

After each labeling session, "freeze" the photos and labels so this exact dataset can always be recovered:

```bash
# Where: Container
dvc add data/raw
git add data/raw.dvc data/.gitignore data/labels.csv
git commit -m "Labels: first round"
dvc push
```

Change the commit message to describe the round, e.g. "Labels: +300 photos, evening lighting".

### 9.4 Optional: cloud backup

To store data versions in the cloud (e.g. an S3-compatible bucket), put the access keys in `.env` using the variable names shown in `.env.example`. Never put them in any other file. Then point DVC at the bucket with `dvc remote add -d cloud s3://BUCKET/PATH`. See the DVC documentation for
your provider.

---

## 10. Train and test the model

### 10.1 Train

```bash
# Where: Container
python -m catwatch.train
```

**First, it prints a table** of how many photos of each type are in each group:

- `train`: photos the model learns from
- `val` (validation): photos it's checked against while learning
- `test`: photos kept aside for a final check

Photos from the same hour always go in the same group. That prevents near-identical photos from being in both learning and testing, which would make results look better than they really are.

If any number in the table is **0**, you need more labeled photos of that type. If `both_cats` has no photos at all, training stops with an error. Fix it by labeling some photos of both cats, or by deleting the `both_cats` lines from `classes` and `label_keys` in `config.yaml` for now.

**Then it trains.** It prints one line per round ("epoch"). The `val macro-F1` number is a score from 0 to 1, and higher is better.

At the end it prints `Saved run to runs/20260921_143015_seed0` (your date and time will differ). This guide calls that folder `RUN_FOLDER`.

### 10.2 Test on the validation photos

```bash
# Where: Container
python -m catwatch.evaluate --run runs/RUN_FOLDER
```

**Reading the results:**

- **precision** for a cat: when the model says "that cat", how often it's right.
- **recall** for a cat: of all the times that cat was really there, how often the model noticed.
- **confusion matrix**: rows are the truth, columns are the model's guess. Numbers off the diagonal are mistakes. The most important cells are one cat mistaken for the other.

**Look at the mistakes yourself.** Open `runs/RUN_FOLDER/errors_val.csv` in VS Code and open some of the listed photos. Usually a mistake is one of these:

- **A labeling error.** Re-run the labeler; to fix an old label, see section 16.
- **A lighting situation with few examples.** Collect more photos at that time of day.
- **A crop that cuts off the cat.** Adjust the crop.

### 10.3 Final test (only once, for the model you'll use)

```bash
# Where: Container
python -m catwatch.evaluate --run runs/RUN_FOLDER --split test
```

Only do this for the model you've decided to deploy. If you keep checking against the test photos while tweaking things, they stop being an honest test.

### 10.4 Comparing two settings fairly (optional, advanced)

One training run can be lucky or unlucky. To check whether a settings change really helps:

1. Train five times with each setting:
   `for s in 0 1 2 3 4; do python -m catwatch.train --seed $s; done`
2. Evaluate each run on `val`.
3. Compare the macro-F1 scores seed by seed.

Only believe the change if it wins by clearly more than the run-to-run wobble: more than 2 × (standard deviation of the differences ÷ √5).

---

## 11. Deploy the model to the Pi

### 11.1 Export the model to the Pi's format

```bash
# Where: Container
python -m catwatch.export --run runs/RUN_FOLDER
```

This converts the model to ONNX, a format the Pi can run quickly. It then checks, on real photos, that the converted model gives the same answers as the original. If the check fails, it stops with "Do not deploy." Look for:

```
ONNX parity check: {'max_abs_logit_difference': ..., 'same_predictions': True} Exported to /workspace/models/RUN_FOLDER
```

### 11.2 Point the settings at the new model

In `config.yaml`, change the `model_dir` line:

```yaml
monitor:
  model_dir: models/RUN_FOLDER
```

Commit it:

```bash
# Where: Container
git add config.yaml
git commit -m "Deploy model RUN_FOLDER"
```

### 11.3 Copy the code and the model to the Pi

First sync code to the Pi (the command in 6.4). Then copy the models:

```bash
# Where: WSL
rsync -av ~/code/cat-bowl-monitor/models/ $PI:~/cat-bowl-monitor/models/
```

### 11.4 Test run on the Pi

Make sure the collector isn't running (section 8.4), then:

```bash
# Where: Pi
cd ~/cat-bowl-monitor
source .venv/bin/activate
python -m catwatch.monitor
```

It prints `Monitoring with model ...`. When a cat drinks for a few seconds and walks away, a line appears, e.g.:

```
14:32:05  Cat with orange patches at bowl for 12s
```

Press **Ctrl+C** to stop.

**Check it against reality.** Watch a real visit and confirm the printed line names the right cat. The photo taken at the start of each visit is also saved, in `data/raw/<date>/..._event-<cat>.jpg`, so you can check visits you
didn't see.

**To see live predictions on the Pi's own screen:** sit at the Pi with the screen and keyboard, open a terminal on its desktop, and run the same commands with `--show` at the end:

```bash
# Where: Pi (terminal on the Pi's own screen, not over ssh)
cd ~/cat-bowl-monitor
source .venv/bin/activate
python -m catwatch.monitor --show
```

---

## 12. Run it automatically at startup

Set this up once so the monitor starts whenever the Pi powers on, and restarts itself if it crashes.

### 12.1 Create the service file

```bash
# Where: Pi
sudo nano /etc/systemd/system/catwatch.service
```

A text editor opens. Paste this in, replacing **all three** `YOURNAME`s with your Pi username:

```ini
[Unit]
Description=Cat water bowl monitor
After=network-online.target

[Service]
User=YOURNAME
WorkingDirectory=/home/YOURNAME/cat-bowl-monitor
ExecStart=/home/YOURNAME/cat-bowl-monitor/.venv/bin/python -m catwatch.monitor
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Save and exit: press **Ctrl+O**, **Enter**, then **Ctrl+X**.

### 12.2 Turn it on

```bash
# Where: Pi
sudo systemctl daemon-reload
sudo systemctl enable --now catwatch
sudo systemctl status catwatch
```

The status should say **active (running)** in green. Press **Q** to get back to the prompt.

### 12.3 Test that it survives a restart

```bash
# Where: Pi
sudo reboot
```

Wait a minute, then reconnect and check:

```bash
# Where: WSL
ssh $PI
```

```bash
# Where: Pi
sudo systemctl status catwatch
```

It should be running again.

---

## 13. Everyday use

### Watch visits as they happen

```bash
# Where: Pi
journalctl -u catwatch -f
```

Press Ctrl+C to stop watching. The monitor keeps running.

### Copy the visit log to your computer

```bash
# Where: WSL
rsync -av $PI:~/cat-bowl-monitor/data/events.csv ~/code/cat-bowl-monitor/data/
```

`events.csv` opens in Excel. Each row is one visit: which cat, start time, end time, and duration in seconds.

**What it measures:** the system detects a cat **at the bowl**, not proof that it drank. A cat sitting by the bowl for several seconds counts as a visit.

### Stop, start, or restart the monitor

```bash
# Where: Pi
sudo systemctl stop catwatch       # stop (e.g. before running collect or check_crop)
sudo systemctl start catwatch      # start again
sudo systemctl restart catwatch    # after copying new code or a new model
```

### Shut down safely

Pulling the power while the Pi is running can damage the SD card. Always shut down first:

```bash
# Where: Pi
sudo shutdown -h now
```

Wait until the Pi's green light stops blinking, then unplug it.

---

## 14. Improving the model over time

The monitor saves the photo at the start of every visit into `data/raw`. These are real-world examples and the most valuable new training data.

1. Pull photos from the Pi (command in 8.5).
2. Label them (section 9.1). The labeler only shows photos without a label. Label what you actually see, not what the model guessed.
3. Save the round as a version (section 9.3).
4. Train (10.1) and evaluate on `val` (10.2).
5. Compare against the current model's scores. Deploy only if it's clearly better (sections 11.1–11.3), then restart the service:
   `sudo systemctl restart catwatch`.

### Going back to an older model

Every deployed model keeps its own folder in `models/`. Point `model_dir` in `config.yaml` back to the old folder, sync code to the Pi, then:

```bash
# Where: Pi
sudo systemctl restart catwatch
```

---

## 15. Changing Wi-Fi networks

### Add a new network before you move the Pi (easiest)

While the Pi is still on the current Wi-Fi, add the new network so it can join it later:

```bash
# Where: Pi
sudo nmcli connection add type wifi con-name "NEW_NAME" ssid "NEW_WIFI_NAME" \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "NEW_WIFI_PASSWORD"
nmcli connection show
```

The new network should appear in the list. The Pi joins whichever saved network is in range when it starts.

### If the Pi is already somewhere it can't connect

Plug in the screen and a USB keyboard and mouse, then either:

- click the **network icon** at the top-right of the Pi desktop and pick the network, or
- open a terminal and run `sudo nmtui`, then choose **Activate a connection**.

### After changing networks

The Pi's IP address will change. Repeat section 5.1 and update your shortcut:

```bash
# Where: WSL
nano ~/.bashrc
```

Scroll to the last line and edit the address in `export PI=...`. Save and exit with Ctrl+O, Enter, Ctrl+X, then run:

```bash
# Where: WSL
source ~/.bashrc
```

Your computer must be on the **same** network as the Pi to connect to it.

---

## 16. Troubleshooting

**`ssh: connect to host ... Connection timed out` or `No route to host`**

- Is the Pi powered on, and did you wait 3–5 minutes after power-on?
- Is your computer on the same Wi-Fi as the Pi?
- Has the IP changed? Repeat section 5.1.
- If it never joined Wi-Fi, connect the screen and keyboard and check the network icon.
- A wrong Wi-Fi password or name in Imager is common. The fix is to connect from the Pi's desktop (section 15), or to re-flash the card.

**`WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED`**

This happens after re-flashing the SD card. Clear the old record:

```bash
# Where: WSL
ssh-keygen -R 192.168.1.57
```

Use your Pi's IP address, then connect again.

**`rpicam-hello` says "No cameras available"**

1. Unplug the power.
2. Re-seat the ribbon at both ends with the contacts facing the correct way (see the diagram link in 4.2), and close the latches.
3. On a Pi 5, confirm you have the 22-pin cable.
4. Power on and try again.

If a guide online says to use `libcamera-hello`, use `rpicam-hello` instead; the command was renamed.

**`error: externally-managed-environment` when running pip on the Pi**

You forgot to activate the project environment. Run
`cd ~/cat-bowl-monitor && source .venv/bin/activate` first.

**Camera errors mentioning "numpy" or "dtype size changed" on the Pi**

A second copy of NumPy got installed. Remove it:

```bash
# Where: Pi (with .venv active)
pip uninstall -y numpy
```

**"Camera in use", "Device or resource busy", or "Failed to acquire camera"**

Another camera program is already running. Stop the service with `sudo systemctl stop catwatch`, and check for a collector in tmux with `tmux ls`.

**The colors in photos look wrong (orange looks blue)**

Stop and report it before labeling anything. Check `crop_check_full.jpg` first.

**Container won't start, with an error mentioning `/mnt/wslg` or `.env`**

- If the error mentions `.env`, create it: `cp .env.example .env` in WSL.
- If the error mentions `/mnt/wslg` or `X11`, your Windows version may not support Linux windows. Remove the two `/tmp/.X11-unix` and `/mnt/wslg` lines from `mounts`, and the whole `containerEnv` block, in `.devcontainer/devcontainer.json`. Then F1 → **Rebuild Container**. Use the labeling workaround below.
- If the error mentions `dvc-storage`, run `mkdir -p ~/dvc-storage` in WSL.

**The labeling window never appears (or errors about "display", "xcb", or "Qt")**

Run the labeler from WSL instead of the container. It only needs a few small packages:

```bash
# Where: WSL
cd ~/code/cat-bowl-monitor
python3 -m venv .venv-label
.venv-label/bin/pip install opencv-python PyYAML python-dotenv
.venv-label/bin/python -m catwatch.label --labeled-by YOURNAME
```

**Key presses do nothing in the labeler**

Click on the photo window first. The keys go to whichever window is selected.

**How do I fix a label I made long ago?**

Never edit `labels.csv` by hand. Ask for help adding a "relabel" command, or note the file name and fix it in a future session. Every correction is kept as a new row, and the newest one wins.

**Training error: "No training images for class index(es) [3]"**

There are no `both_cats` photos in the training group. See section 10.1.

**Training crashes with "bus error" or mentions "shared memory"**

Rebuild the container (F1 → **Rebuild Container**). If it still happens, set
`num_workers: 0` under `training` in `config.yaml`.

**Scores are nearly perfect right away**

Be suspicious. Check the table printed at the start of training. With only a few days of data, the `val` group may only contain one or two times of day. Collect more varied data before trusting it.

**Everything in the container is very slow**

The project is probably under `/mnt/c/...`. Move it to `~/code` inside WSL
(section 3.5).

**Git says "dubious ownership"**

Run `id -u` in WSL. If it isn't `1000`, the container user ID in `Dockerfile` needs to match it.

**The Pi gets hot or shows `throttled` values other than `0x0`**

- Heat: add ventilation or a small fan to the case.
- Power: undervoltage means the supply is too weak. Use the official power supply.

**Visits are split in two, or brief walk-bys are logged**

In `config.yaml` under `monitor`, adjust these, then sync code and restart the service:

- `min_event_seconds` (a higher number ignores short visits)
- `smoothing_window` (keep it an odd number)

Check the effect against real observations, not just the log.

**Something else is wrong**

Copy the **entire** error message, from the first line of the traceback to the last, and ask for help with it.

---

## 17. Command cheat sheet

| Task | Where | Command |
|---|---|---|
| Log into the Pi | WSL | `ssh $PI` |
| Leave the Pi | Pi | `exit` |
| Activate the project on the Pi | Pi | `cd ~/cat-bowl-monitor && source .venv/bin/activate` |
| Sync code to the Pi | WSL | see section 6.4 |
| Check the crop | Pi | `python -m catwatch.check_crop` |
| Collect photos | Pi | `tmux new -s collect`, then `python -m catwatch.collect`, then Ctrl+B, D |
| Pull photos from the Pi | WSL | `rsync -av $PI:~/cat-bowl-monitor/data/raw/ ~/code/cat-bowl-monitor/data/raw/` |
| Label | Container | `python -m catwatch.label --labeled-by YOURNAME` |
| Save a data version | Container | `dvc add data/raw && git add data/raw.dvc data/.gitignore data/labels.csv && git commit -m "Labels: ..." && dvc push` |
| Train | Container | `python -m catwatch.train` |
| Evaluate | Container | `python -m catwatch.evaluate --run runs/RUN_FOLDER` |
| Export | Container | `python -m catwatch.export --run runs/RUN_FOLDER` |
| Copy models to the Pi | WSL | `rsync -av ~/code/cat-bowl-monitor/models/ $PI:~/cat-bowl-monitor/models/` |
| Restart the monitor | Pi | `sudo systemctl restart catwatch` |
| Watch visits live | Pi | `journalctl -u catwatch -f` |
| Get the visit log | WSL | `rsync -av $PI:~/cat-bowl-monitor/data/events.csv ~/code/cat-bowl-monitor/data/` |
| Check code quality | Container | `make verify` |
| Shut down the Pi | Pi | `sudo shutdown -h now` |