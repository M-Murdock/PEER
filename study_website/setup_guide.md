# Step-by-Step Setup Guide: Pygame to Web

## Quick Overview
1. Install Pygbag
2. Make your Pygame code web-ready
3. Convert to web format
4. Integrate into HTML
5. Test and deploy

---

## Step 1: Install Pygbag

Open your terminal and run:

```bash
pip install pygbag
```

---

## Step 2: Make Your Pygame Code Web-Ready

Your Pygame code needs two modifications:

### A) Add async/await to your main loop

**Before:**
```python
running = True
while running:
    for event in pygame.event.get():
        # handle events
    # game logic
    pygame.display.flip()
```

**After:**
```python
import asyncio

async def main():
    running = True
    while running:
        for event in pygame.event.get():
            # handle events
        # game logic
        pygame.display.flip()
        await asyncio.sleep(0)  # CRITICAL!

asyncio.run(main())
```

### B) (Optional) Send results to HTML

Add this function to send data back to your study page:

```python
import sys

def send_results():
    if sys.platform == "emscripten":  # Running in browser
        try:
            from browser import window
            results = {
                "type": "simulator_complete",
                "score": your_score,
                "time": your_time,
                # add whatever data you want
            }
            window.parent.postMessage(results, "*")
        except:
            pass
```

---

## Step 3: Convert to Web Format

### Create your project structure:
```
my_simulator/
├── main.py          # Your Pygame code
└── assets/          # Any images, sounds, etc.
    ├── image.png
    └── sound.wav
```

### Run Pygbag:

```bash
cd my_simulator
pygbag .
```

This creates:
```
my_simulator/
├── main.py
├── assets/
└── build/
    └── web/
        ├── index.html      # ← This is what you need!
        ├── main.py
        └── ... (other files)
```

---

## Step 4: Integrate into Your Study Website

### A) Set up your folder structure:

```
study_website/
├── study_framework.html
└── simulators/
    ├── simulator1/
    │   └── (copy everything from build/web here)
    └── simulator2/
        └── (copy everything from build/web here)
```

### B) Update the HTML:

Find this section in your HTML:

```html
<div class="simulator-placeholder">
    [Pygame Simulator 1 Will Load Here]
</div>
```

Replace it with:

```html
<iframe 
    src="simulators/simulator1/index.html" 
    width="800" 
    height="600"
    frameborder="0"
    style="border: none;">
</iframe>
```

---

## Step 5: Test Locally

### Option A: Using Python's HTTP server

```bash
cd study_website
python -m http.server 8000
```

Then open: http://localhost:8000/study_framework.html

### Option B: Using VS Code Live Server

1. Install "Live Server" extension in VS Code
2. Right-click on `study_framework.html`
3. Select "Open with Live Server"

---

## Step 6: Deploy

### Upload to web host
- Your university server
- GitHub Pages
- Netlify
- Vercel

Make sure to upload the entire folder structure:
```
study_website/
├── study_framework.html
└── simulators/
    ├── simulator1/ (all files)
    └── simulator2/ (all files)
```

---

## Troubleshooting

### "Module not found" errors
- Make sure all your imports are at the top of main.py
- Pygbag includes: pygame, asyncio, random, math, json
- For other modules, you may need workarounds

### Simulator not loading
- Check browser console (F12) for errors
- Make sure file paths are correct
- Test with a simple example first

### Data not being captured
- Check browser console to see if messages are being received
- Make sure `window.addEventListener('message', ...)` is in your HTML
- Test with: `console.log('Received:', event.data)`

---

## Example File Tree (Final)

```
study_website/
├── study_framework.html          # Main page
├── simulators/
│   ├── simulator1/
│   │   ├── index.html           # From pygbag
│   │   ├── main.py
│   │   └── ... (other pygbag files)
│   └── simulator2/
│       ├── index.html
│       ├── main.py
│       └── ...
└── README.md                     # Optional documentation
```

---

## Testing Checklist

- [ ] Simulator loads in iframe
- [ ] Can interact with simulator (mouse/keyboard)
- [ ] Simulator completes and sends data
- [ ] "Continue" button works
- [ ] Data appears in browser console
- [ ] Form validation works
- [ ] All pages navigate correctly
- [ ] Works in different browsers (Chrome, Firefox, Safari)

---

## Need Help?

If you run into issues:
1. Check the browser console (F12)
2. Look at the example_simulator.py for reference
3. Test with a minimal Pygame example first
4. Make sure your server is serving all files correctly
