## Colab-Compatible Resumable Notebook Pattern

All notebooks intended to run in Google Colab OR locally must follow this pattern.

### 1. Environment detection (always first cell)
```python
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
```

### 2. Path setup — branch on IN_COLAB
- Colab: mount Drive, root at `Path('/content/drive/MyDrive/ham10000-augmentation')`
- Local: root at `Path.cwd()` (notebook launched from project root)
- Never hardcode absolute paths; all paths derive from the root variable

### 3. Dependency installation — try/except pattern
```python
try:
    import some_package
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'some_package'])
    import some_package
```

### 4. Completion markers — skip already-done work
Every major block must check for its output artifact before running:
```python
if (run_dir / 'test_metrics.json').exists():
    print('Ya completado — saltando')
    # load and return existing result
```
Completion marker = the primary output artifact of that block (e.g., `test_metrics.json`, `embedding_final.pt`, `*.jpg` count).

### 5. Checkpointing — resume across sessions
For long-running loops (training epochs, image generation):
- Save `checkpoint_last.pt` (or `checkpoint_last.json`) after every unit of work (epoch, batch of images)
- Checkpoint must include: current position + all state needed to continue (model weights, optimizer, scheduler, history, best metric so far)
- On start: if checkpoint exists AND completion marker absent → load checkpoint and resume
```python
if ckpt_path.exists() and not done_marker.exists():
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch'] + 1
```

### 6. Drive output structure (Colab)
```
Mi unidad/ham10000-augmentation/
  experiments/<scenario>/
    best_model.pt          ← best val F1 checkpoint
    checkpoint_last.pt     ← resume checkpoint (deleted after test_metrics.json saved)
    history.json           ← per-epoch metrics
    test_metrics.json      ← COMPLETION MARKER
    confusion_matrix.png
    roc_curve.png
  synthetic/
    textual_inversion/
    img2img/
```

### 7. Progress cell — run without loading models
Include a cell near the top that shows completion status of all blocks without requiring model loading. Allows quick status check after reconnecting.

### 8. Self-contained code
Notebooks must not import from `scripts/` using relative paths — those use `Path(__file__).resolve()` which breaks in Colab. Inline all necessary logic or copy it explicitly.

### 9. Hardware adaptation
- Device: detect CUDA → MPS → CPU
- dtype: `float16` for CUDA, `float32` for MPS/CPU
- batch_size: scale with VRAM (e.g., T4 16GB→32, CPU→8)
- num_workers: 2 for CUDA, 0 for MPS/CPU
