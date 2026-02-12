# Guide: Integrating Pygame Simulators into Your Study Website

## Option 1: Pygbag (Pygame for Web) - RECOMMENDED

Pygbag converts your Pygame code to run directly in the browser.

### Step 1: Install Pygbag
```bash
pip install pygbag
```

### Step 2: Prepare Your Pygame Code
Create your simulator as `main.py`:

```python
import pygame
import asyncio

# Your pygame code here
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

async def main():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Your game logic here
        screen.fill((255, 255, 255))
        
        pygame.display.flip()
        clock.tick(60)
        await asyncio.sleep(0)  # CRITICAL for web

asyncio.run(main())
```

**CRITICAL**: You MUST use `async/await` and `asyncio.sleep(0)` in your game loop for web compatibility.

### Step 3: Build for Web
```bash
pygbag your_game_folder
```

This creates a web-ready version in the `build/web` directory.

### Step 4: Integrate into HTML
Replace the simulator placeholder with an iframe:

```html
<div class="simulator-container">
    <iframe 
        src="simulators/simulator1/index.html" 
        width="800" 
        height="600"
        frameborder="0"
        style="border: none;">
    </iframe>
</div>
```

### Folder Structure:
```
study_website/
├── study_framework.html
└── simulators/
    ├── simulator1/
    │   └── index.html (from pygbag build)
    └── simulator2/
        └── index.html (from pygbag build)
```

---

## Option 2: Brython (Python in Browser)

Run Python code directly in the browser using Brython (no Pygame graphics, but can do basic interactions).

```html
<script src="https://cdn.jsdelivr.net/npm/brython@3/brython.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/brython@3/brython_stdlib.js"></script>

<body onload="brython()">
    <script type="text/python">
        # Your Python code here
        from browser import document, html
        
        document <= html.H1("Hello from Python!")
    </script>
</body>
```

---

## Option 3: Convert to JavaScript

Rewrite your Pygame logic using HTML5 Canvas and JavaScript:

```html
<canvas id="gameCanvas" width="800" height="600"></canvas>
<script>
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');
    
    function gameLoop() {
        // Clear canvas
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Your game logic here
        ctx.fillStyle = 'blue';
        ctx.fillRect(100, 100, 50, 50);
        
        requestAnimationFrame(gameLoop);
    }
    
    gameLoop();
</script>
```

---

## Option 4: Backend Server with VNC/Screenshots

Run Pygame on a server and stream it to the browser (complex, not recommended for studies).

---

## RECOMMENDED APPROACH FOR YOUR STUDY:

**Use Pygbag** because:
- ✅ Keep your existing Pygame code
- ✅ Runs directly in browser (no installation needed)
- ✅ Works on all devices
- ✅ Easy to integrate with your HTML framework

---

## Communication Between Pygame and HTML

### From Pygame to HTML (send data):
```python
# In your Pygame code
import platform
if platform.system() == "Emscripten":
    # Running in browser
    from browser import window
    window.parent.postMessage({"score": 100, "completed": True}, "*")
```

### From HTML to Pygame (receive data):
```javascript
// In your HTML
window.addEventListener('message', function(event) {
    console.log('Data from Pygame:', event.data);
    // Store in studyData
    studyData.simulator1Results = event.data;
});
```

---

## Quick Start Example

I'll create a complete working example in the next file showing:
1. A simple Pygame simulator
2. How to convert it with Pygbag
3. How to integrate it into your study framework
