#!/usr/bin/env python3
"""
Song Generator - Web GUI for YuE Music Generation Model
A Flask-based web application that wraps the YuE model for generating songs from lyrics.
"""

import os
import sys
import json
import uuid
import time
import shutil
import subprocess
import threading
import signal
import atexit
import re
import glob
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
from flask import Flask, render_template, request, jsonify, send_file, Response, send_from_directory
import queue

# ─── Configuration ───────────────────────────────────────────────────────────

# Paths
BASE_DIR = Path(__file__).resolve().parent
YUE_DIR = Path("/mnt/nas10_shared/jaden/YuE")
YUE_INFERENCE_DIR = YUE_DIR / "inference"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"

# Create necessary directories
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Available models
MODELS = OrderedDict({
    "english_cot": {
        "name": "English (CoT - No Reference)",
        "stage1": "m-a-p/YuE-s1-7B-anneal-en-cot",
        "stage2": "m-a-p/YuE-s2-1B-general",
        "description": "English song generation without reference audio",
        "icl": False,
    },
    "english_icl": {
        "name": "English (ICL - With Reference)",
        "stage1": "m-a-p/YuE-s1-7B-anneal-en-icl",
        "stage2": "m-a-p/YuE-s2-1B-general",
        "description": "English song generation with reference audio for style guidance",
        "icl": True,
    },
    "chinese_cot": {
        "name": "Chinese / Mandarin (CoT)",
        "stage1": "m-a-p/YuE-s1-7B-anneal-zh-cot",
        "stage2": "m-a-p/YuE-s2-1B-general",
        "description": "Chinese/Mandarin song generation without reference audio",
        "icl": False,
    },
    "chinese_icl": {
        "name": "Chinese / Mandarin (ICL)",
        "stage1": "m-a-p/YuE-s1-7B-anneal-zh-icl",
        "stage2": "m-a-p/YuE-s2-1B-general",
        "description": "Chinese/Mandarin song generation with reference audio",
        "icl": True,
    },
    "japanese_cot": {
        "name": "Japanese / Korean (CoT)",
        "stage1": "m-a-p/YuE-s1-7B-anneal-jp-kr-cot",
        "stage2": "m-a-p/YuE-s2-1B-general",
        "description": "Japanese/Korean song generation without reference audio",
        "icl": False,
    },
    "japanese_icl": {
        "name": "Japanese / Korean (ICL)",
        "stage1": "m-a-p/YuE-s1-7B-anneal-jp-kr-icl",
        "stage2": "m-a-p/YuE-s2-1B-general",
        "description": "Japanese/Korean song generation with reference audio",
        "icl": True,
    },
})

# Top genre/style tags from YuE
POPULAR_TAGS = [
    "Pop", "Rock", "R&B", "Electronic", "Folk", "Jazz", "Blues", "Country",
    "Hip-hop", "Rap", "Classical", "Ambient", "punk", "indie-rock",
    "K-pop", "Dance", "Disco", "House", "Funk", "Soul", "Reggae",
    "ballad", "acoustic", "lofi", "synthwave", "orchestral",
    "uplifting", "inspiring", "melancholic", "energetic", "dreamy",
    "dark", "romantic", "happy", "sad", "epic", "chill", "groovy",
    "female", "male", "airy", "powerful", "soft", "vocal",
    "guitar", "piano", "strings", "drums", "bass",
]

# Allowed extensions for reference audio
ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}


# ─── Flask Application ───────────────────────────────────────────────────────

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

# Job queue for background processing
job_queue = queue.Queue()
jobs = {}  # job_id -> job_info dict
jobs_lock = threading.Lock()
active_inference_process = None
gpu_busy = False  # Only one generation at a time per GPU
gpu_lock = threading.Lock()


def cleanup_on_exit():
    """Clean up any running inference processes on exit."""
    global active_inference_process
    if active_inference_process and active_inference_process.poll() is None:
        active_inference_process.terminate()
        print("[cleanup] Terminated running inference process.")


atexit.register(cleanup_on_exit)


# ─── Helper Functions ────────────────────────────────────────────────────────

def ensure_lyrics_structure(lyrics_text):
    """
    Ensure lyrics have at least 2 structure sections (verse + chorus).
    YuE's split_lyrics() requires structure tags to work, and infer.py
    needs at least 2 segments (run_n_segments >= 2) to avoid the
    'raw_output is not defined' bug.
    """
    valid_tags = {'verse', 'chorus', 'bridge', 'outro', 'intro', 
                  'pre-chorus', 'hook', 'interlude', 'solo', 'refrain'}
    
    # Find existing structure tags
    found_tags = re.findall(r'\[(\w+(?:-\w+)?)\]', lyrics_text, re.IGNORECASE)
    found_tags_lower = [t.lower() for t in found_tags if t.lower() in valid_tags]
    
    if len(found_tags_lower) >= 2:
        # Already has 2+ sections — good to go
        return lyrics_text
    
    # Need to fix: either no tags or only 1 section
    lines = [l.strip() for l in lyrics_text.strip().split('\n') if l.strip()]
    
    if len(found_tags_lower) == 1:
        # Has 1 tag — add a chorus after the existing content
        tag = found_tags_lower[0]
        if tag == 'chorus':
            return lyrics_text.strip() + "\n\n[verse]\nKeep the rhythm going strong\nWe can sing all night long"
        else:
            return lyrics_text.strip() + "\n\n[chorus]\nThis is our song, we sing along\nTogether we belong\nOur melody carries on"
    
    # No tags at all — auto-structure
    if not lines:
        return "[verse]\nLa la la, sing a song\nA melody all day long\n\n[chorus]\nThis is the chorus now\nWe'll make it through somehow"
    
    # Split lines into two halves
    mid = max(1, len(lines) // 2)
    parts = [
        "[verse]",
        *lines[:mid],
        "",
        "[chorus]",
        *lines[mid:],
    ]
    return '\n'.join(parts)


def infer_song(job_id, lyrics_raw, genre_tags, model_key, 
               audio_prompt_path=None, prompt_start=0, prompt_end=30,
               dual_vocal_path=None, dual_inst_path=None,
               run_n_segments=2, max_new_tokens=3000, 
               repetition_penalty=1.1, stage2_batch_size=4, seed=42,
               cuda_idx=0):
    """
    Runs the YuE inference as a subprocess.
    Updates the job status dict as it progresses.
    """
    global active_inference_process
    
    with jobs_lock:
        jobs[job_id]['status'] = 'preparing'
        jobs[job_id]['progress'] = 5
    
    # Create job-specific working directory
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    
    # Auto-structure lyrics if no tags present
    lyrics_raw = ensure_lyrics_structure(lyrics_raw)
    
    # Write genre file
    genre_path = job_dir / "genre.txt"
    with open(genre_path, 'w') as f:
        f.write(genre_tags.strip())
    
    # Write lyrics file
    lyrics_path = job_dir / "lyrics.txt"
    with open(lyrics_path, 'w') as f:
        f.write(lyrics_raw.strip())
    
    # Count lyric segments — ensure at least 2 for the infer.py loop
    segments = re.findall(r'\[(\w+)\]', lyrics_raw)
    actual_segments = max(2, min(run_n_segments, len(segments)) if segments else run_n_segments)
    
    model_info = MODELS[model_key]
    
    # Build command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        str(YUE_INFERENCE_DIR / "infer.py"),
        "--cuda_idx", str(cuda_idx),
        "--stage1_model", model_info["stage1"],
        "--stage2_model", model_info["stage2"],
        "--genre_txt", str(genre_path),
        "--lyrics_txt", str(lyrics_path),
        "--run_n_segments", str(actual_segments),
        "--stage2_batch_size", str(stage2_batch_size),
        "--output_dir", str(job_dir),
        "--max_new_tokens", str(max_new_tokens),
        "--repetition_penalty", str(repetition_penalty),
        "--seed", str(seed),
    ]
    
    # Add reference audio if provided (single track ICL)
    if audio_prompt_path and model_info["icl"]:
        cmd.extend([
            "--use_audio_prompt",
            "--audio_prompt_path", str(audio_prompt_path),
            "--prompt_start_time", str(prompt_start),
            "--prompt_end_time", str(prompt_end),
        ])
    
    # Add dual track reference if provided
    if dual_vocal_path and dual_inst_path and model_info["icl"]:
        cmd.extend([
            "--use_dual_tracks_prompt",
            "--vocal_track_prompt_path", str(dual_vocal_path),
            "--instrumental_track_prompt_path", str(dual_inst_path),
            "--prompt_start_time", str(prompt_start),
            "--prompt_end_time", str(prompt_end),
        ])
    
    with jobs_lock:
        jobs[job_id]['status'] = 'generating'
        jobs[job_id]['progress'] = 10
        jobs[job_id]['command'] = ' '.join(cmd)
    
    try:
        # Set environment for the subprocess
        env = os.environ.copy()
        env['PYTHONPATH'] = str(YUE_INFERENCE_DIR) + ':' + env.get('PYTHONPATH', '')
        # NOTE: Do NOT set CUDA_VISIBLE_DEVICES here — infer.py uses --cuda_idx directly.
        # Setting CUDA_VISIBLE_DEVICES would remap GPU indices and cause "invalid device ordinal".
        
        process = subprocess.Popen(
            cmd,
            cwd=str(YUE_INFERENCE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        active_inference_process = process
        
        output_lines = []
        stage1_done = False
        stage2_done = False
        
        for line in process.stdout:
            line = line.strip()
            if line:
                output_lines.append(line)
                # Parse progress from tqdm or stage indicators
                if "Stage 1 inference" in line or "Stage1" in line:
                    with jobs_lock:
                        jobs[job_id]['progress'] = 15
                elif "100%" in line and not stage1_done:
                    stage1_done = True
                    with jobs_lock:
                        jobs[job_id]['progress'] = 50
                        jobs[job_id]['status'] = 'stage2'
                elif "Stage 2 inference" in line or "Stage2" in line:
                    with jobs_lock:
                        jobs[job_id]['progress'] = 55
                        jobs[job_id]['status'] = 'stage2'
                elif "100%" in line and stage1_done and not stage2_done:
                    stage2_done = True
                    with jobs_lock:
                        jobs[job_id]['progress'] = 95
                # Track tqdm percentages
                pct_match = re.search(r'(\d+)%', line)
                if pct_match:
                    pct = int(pct_match.group(1))
                    if not stage1_done:
                        mapped_pct = 15 + int(pct * 0.35)  # 15-50%
                        with jobs_lock:
                            jobs[job_id]['progress'] = max(jobs[job_id]['progress'], mapped_pct)
                    elif not stage2_done:
                        mapped_pct = 55 + int(pct * 0.40)  # 55-95%
                        with jobs_lock:
                            jobs[job_id]['progress'] = max(jobs[job_id]['progress'], mapped_pct)
        
        process.wait()
        active_inference_process = None
        
        if process.returncode != 0:
            with jobs_lock:
                jobs[job_id]['status'] = 'error'
                jobs[job_id]['error'] = '\n'.join(output_lines[-20:])  # Last 20 lines
            return
        
        # Find the generated audio files
        # YuE outputs mp3 to the root output dir, recons/, and vocoder/ subdirs
        audio_files = []
        # Search entire job directory for audio files, prefer mixed/final
        for ext in ['*.mp3', '*.wav', '*.flac']:
            audio_files.extend(glob.glob(str(job_dir / '**' / ext), recursive=True))
        
        # Sort: prefer "mixed" files at root level, then others
        def sort_key(f):
            name = os.path.basename(f)
            is_mixed = 'mixed' in name.lower()
            is_root = os.path.dirname(f) == str(job_dir)
            is_recons = 'recons' in f
            return (not is_mixed, not is_root, is_recons, name)
        
        audio_files.sort(key=sort_key)
        
        with jobs_lock:
            jobs[job_id]['status'] = 'done'
            jobs[job_id]['progress'] = 100
            jobs[job_id]['output_files'] = audio_files
            jobs[job_id]['output_dir'] = str(job_dir)
            jobs[job_id]['completed_at'] = datetime.now().isoformat()
        
    except Exception as e:
        with jobs_lock:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = str(e)


def background_worker():
    """Background thread that processes jobs from the queue one at a time."""
    global gpu_busy
    while True:
        try:
            job = job_queue.get(timeout=1)
            if job is None:
                break
            # Acquire GPU lock — only one generation at a time
            with gpu_lock:
                gpu_busy = True
                try:
                    # Update any queued jobs to show they're waiting
                    with jobs_lock:
                        for jid, j in jobs.items():
                            if j['status'] == 'queued' and jid != job['job_id']:
                                j['status'] = 'waiting'
                    
                    infer_song(**job)
                finally:
                    gpu_busy = False
                    # Small delay to let GPU fully release
                    import time
                    time.sleep(2)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[worker] Error processing job: {e}")
            with jobs_lock:
                gpu_busy = False


# Start background worker thread
worker_thread = threading.Thread(target=background_worker, daemon=True)
worker_thread.start()


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Render the main song generator page."""
    return render_template('index.html', models=MODELS, tags=POPULAR_TAGS)


@app.route('/api/models')
def get_models():
    """Return available models."""
    return jsonify(MODELS)


@app.route('/api/tags')
def get_tags():
    """Return popular genre/style tags."""
    return jsonify(POPULAR_TAGS)


@app.route('/api/generate', methods=['POST'])
def generate_song():
    """
    Submit a song generation job.
    Accepts multipart/form-data with: lyrics, genre, model, audio_prompt (optional),
    segments, max_tokens, repetition_penalty, batch_size, seed.
    """
    # Read from form data (frontend sends multipart/form-data via FormData)
    lyrics = request.form.get('lyrics', '').strip()
    if not lyrics:
        return jsonify({"error": "Lyrics are required. Please enter song lyrics."}), 400
    
    genre = request.form.get('genre', 'pop uplifting melodic female airy vocal').strip()
    model_key = request.form.get('model', 'english_cot')
    if model_key not in MODELS:
        return jsonify({"error": f"Invalid model: {model_key}"}), 400
    
    # Create job
    job_id = str(uuid.uuid4())[:8]
    
    with jobs_lock:
        # If GPU is busy, show waiting status
        initial_status = 'waiting' if gpu_busy else 'queued'
        jobs[job_id] = {
            'id': job_id,
            'status': initial_status,
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'lyrics': lyrics[:200] + '...' if len(lyrics) > 200 else lyrics,
            'genre': genre,
            'model': model_key,
            'output_files': [],
            'output_dir': None,
            'error': None,
        }
    
    # Handle reference audio if included in multipart upload
    audio_prompt_path = None
    prompt_start = float(request.form.get('prompt_start', 0))
    prompt_end = float(request.form.get('prompt_end', 30))
    
    # Check for uploaded reference audio file
    if 'audio_prompt' in request.files:
        audio_file = request.files['audio_prompt']
        if audio_file.filename:
            ext = Path(audio_file.filename).suffix.lower()
            if ext not in ALLOWED_AUDIO_EXTENSIONS:
                return jsonify({"error": f"Invalid audio format: {ext}. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"}), 400
            ref_path = UPLOAD_DIR / f"ref_{job_id}{ext}"
            audio_file.save(str(ref_path))
            audio_prompt_path = str(ref_path)
    
    # Build job args
    job_args = {
        'job_id': job_id,
        'lyrics_raw': lyrics,
        'genre_tags': genre,
        'model_key': model_key,
        'audio_prompt_path': audio_prompt_path,
        'prompt_start': prompt_start,
        'prompt_end': prompt_end,
        'run_n_segments': int(request.form.get('segments', 2)),
        'max_new_tokens': int(request.form.get('max_tokens', 3000)),
        'repetition_penalty': float(request.form.get('repetition_penalty', 1.1)),
        'stage2_batch_size': int(request.form.get('batch_size', 4)),
        'seed': int(request.form.get('seed', 42)),
        'cuda_idx': int(request.form.get('cuda_idx', 0)),
    }
    
    # Put job in queue
    job_queue.put(job_args)
    
    if gpu_busy:
        msg = '⏳ A song is currently being generated. Yours will start right after!'
    else:
        msg = 'Your song is being generated! 🎵'
    
    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'message': msg,
    })


@app.route('/api/status/<job_id>')
def job_status(job_id):
    """Get the status of a generation job."""
    with jobs_lock:
        job = jobs.get(job_id)
    
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify({
        'job_id': job['id'],
        'status': job['status'],
        'progress': job['progress'],
        'error': job.get('error'),
        'has_output': len(job.get('output_files', [])) > 0,
    })


@app.route('/api/download/<job_id>')
def download_song(job_id):
    """Download the generated song."""
    with jobs_lock:
        job = jobs.get(job_id)
    
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    output_files = job.get('output_files', [])
    if not output_files:
        return jsonify({"error": "No output files available yet"}), 404
    
    # Find the best audio file (prefer combined/final, then mp3, then wav)
    mp3_files = [f for f in output_files if f.endswith('.mp3')]
    wav_files = [f for f in output_files if f.endswith('.wav')]
    
    target_file = None
    # Prefer files that look like final combined output
    for f in mp3_files + wav_files:
        if 'combined' in f.lower() or 'final' in f.lower() or 'mix' in f.lower():
            target_file = f
            break
    
    if not target_file:
        target_file = (mp3_files or wav_files or output_files)[0]
    
    return send_file(target_file, as_attachment=True, download_name=f"song_{job_id}.mp3")


@app.route('/api/listen/<job_id>')
def listen_song(job_id):
    """Stream the generated song for in-browser listening."""
    with jobs_lock:
        job = jobs.get(job_id)
    
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    output_files = job.get('output_files', [])
    if not output_files:
        return jsonify({"error": "No output files available yet"}), 404
    
    # Find the best audio file
    mp3_files = [f for f in output_files if f.endswith('.mp3')]
    wav_files = [f for f in output_files if f.endswith('.wav')]
    
    target_file = None
    for f in mp3_files + wav_files:
        if 'combined' in f.lower() or 'final' in f.lower() or 'mix' in f.lower():
            target_file = f
            break
    
    if not target_file:
        target_file = (mp3_files or wav_files or output_files)[0]
    
    return send_file(target_file, mimetype='audio/mpeg' if target_file.endswith('.mp3') else 'audio/wav')


@app.route('/api/jobs')
def list_jobs():
    """List all recent jobs."""
    with jobs_lock:
        job_list = []
        for jid, job in jobs.items():
            job_list.append({
                'id': jid,
                'status': job['status'],
                'progress': job['progress'],
                'lyrics': job.get('lyrics', ''),
                'genre': job.get('genre', ''),
                'created_at': job.get('created_at', ''),
                'has_output': len(job.get('output_files', [])) > 0,
            })
    
    # Sort by created_at descending
    job_list.sort(key=lambda x: x['created_at'], reverse=True)
    return jsonify(job_list[:20])  # Return last 20 jobs


@app.route('/api/progress/<job_id>')
def progress_stream(job_id):
    """SSE endpoint for real-time progress updates."""
    def generate():
        last_progress = -1
        last_status = ''
        while True:
            with jobs_lock:
                job = jobs.get(job_id)
            
            if not job:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break
            
            current_progress = job['progress']
            current_status = job['status']
            
            if current_progress != last_progress or current_status != last_status:
                yield f"data: {json.dumps({'progress': current_progress, 'status': current_status, 'error': job.get('error')})}\n\n"
                last_progress = current_progress
                last_status = current_status
            
            if current_status in ('done', 'error'):
                break
            
            time.sleep(2)
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory(STATIC_DIR, filename)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                 🎵 Song Generator - YuE 🎵                   ║
║                                                              ║
║  Open your browser to: http://localhost:{port}                ║
║                                                              ║
║  Models available:                                           ║""")
    for k, v in MODELS.items():
        print(f"║    • {v['name']}")
    print("""║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
