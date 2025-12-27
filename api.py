from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import torch
import numpy as np
from scipy.io import wavfile
from tortoise.api_fast import TextToSpeech
from tortoise.utils.audio import load_audio
import time
import os
import platform
import logging
import asyncio
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from logging.handlers import RotatingFileHandler

app = FastAPI()

# Configure logging

# Create logs directory if it doesn't exist
log_dir = Path(os.environ.get('LOG_DIR', ''))
log_dir.mkdir(exist_ok=True)

# Configure logging with both file and console output
file_handler = RotatingFileHandler(
    log_dir / 'tortoise_tts_api.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        file_handler,
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global concurrency control - serialize GPU access
gpu_semaphore = asyncio.Semaphore(1)  # serialize GPU use; raise to 2 only if stable
gpu_lock = threading.Lock()  # for synchronous operations in streaming

# Global error state tracking for CUDA failures
cuda_error_state = {
    "has_cuda_error": False,
    "last_error_time": None,
    "error_count": 0,
    "last_error_message": None
}

# Initialize TTS model with conservative settings for stability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Enable DeepSpeed on Linux, disable on Windows for compatibility
is_linux = platform.system() == 'Linux'
use_deepspeed = is_linux and torch.cuda.is_available()

print(f"Platform: {platform.system()}")
print(f"DeepSpeed: {'Enabled' if use_deepspeed else 'Disabled'}")

# Conservative settings for production stability
tts = TextToSpeech(
    use_deepspeed=use_deepspeed,   # Enable on Linux, disable on Windows
    kv_cache=True,                 # 5x faster according to changelog  
    half=True,                     # Half precision for speed and memory
    autoregressive_batch_size=4,   # Reduced from 16 to 4 for stability
    device=device
)

# Voice sample caching to avoid repeated file I/O
@lru_cache(maxsize=64)
def _cached_voice_samples(voice_name: str):
    """Cache voice samples to avoid repeated file I/O and reduce race conditions."""
    if voice_name == "random":
        return None
    
    logger.info(f"Loading voice samples for '{voice_name}' (cache miss)")
    try:
        voice_samples = load_voice_samples(voice_name)
        if not voice_samples:
            logger.error(f"load_voice_samples returned empty list for '{voice_name}'")
            raise ValueError(f"No voice samples loaded for '{voice_name}'")
        logger.info(f"Successfully cached {len(voice_samples)} voice samples for '{voice_name}'")
        return voice_samples
    except Exception as e:
        logger.error(f"Failed to load voice samples for '{voice_name}': {e}")
        raise

def clear_voice_cache():
    """Clear the voice samples cache. Useful for debugging voice loading issues."""
    _cached_voice_samples.cache_clear()
    logger.info("Voice samples cache cleared")

def ensure_vram(min_free_bytes=1_000_000_000):  # ~1 GB
    """Check if there's enough VRAM headroom before starting inference."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        free = total - reserved
        return free >= min_free_bytes
    return True

class SynthesizePayload(BaseModel):
    text: str
    voice: str = "random"
    preset: str = "ultra_realtime"  # Default to ultra_realtime for <500ms target

def get_preset_settings(preset):
    """
    Returns the settings for a given preset, compatible with tts_stream method.
    Conservative settings for production stability.
    """
    # Base settings for all presets
    settings = {
        'temperature': 0.8, 
        'length_penalty': 1.0, 
        'repetition_penalty': 2.0,
        'top_p': 0.8,
        'cond_free_k': 2.0, 
        'diffusion_temperature': 1.0,
        'cond_free': True,
        'k': 1,
        'verbose': False  # Reduce console output
    }
    
    # Preset-specific settings - conservative for stability
    presets = {
        'ultra_realtime': {
            'num_autoregressive_samples': 1, 
            'diffusion_iterations': 1,  # Absolute minimum for <500ms
            'cond_free': False,  # Disable conditioning-free for speed
            'temperature': 0.7,  # Slightly lower for more deterministic output
        },
        'ultra_fast': {
            'num_autoregressive_samples': 1, 
            'diffusion_iterations': 8  # Reduced from 10
        },
        'fast': {
            'num_autoregressive_samples': 16,  # Reduced from 32
            'diffusion_iterations': 25  # Reduced from 50
        },
        'standard': {
            'num_autoregressive_samples': 128,  # Reduced from 256
            'diffusion_iterations': 100  # Reduced from 200
        },
        'high_quality': {
            'num_autoregressive_samples': 128,  # Reduced from 256
            'diffusion_iterations': 200  # Reduced from 400
        },
    }
    
    # Update with preset-specific settings
    if preset in presets:
        settings.update(presets[preset])
    else:
        # Default to ultra_realtime for unknown presets
        settings.update(presets['ultra_realtime'])
    
    return settings

@contextmanager
def cuda_error_recovery():
    """
    Context manager to handle CUDA errors and track recovery failures.
    When a CUDA error occurs and recovery fails, it updates the global error state.
    """
    global cuda_error_state
    try:
        yield
    except Exception as e:
        # Check if this is a CUDA-related error
        error_str = str(e).lower()
        if 'cuda' in error_str or 'gpu' in error_str or torch.cuda.is_available():
            try:
                # Attempt CUDA recovery
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                logger.info("CUDA recovery attempted")
                # If we get here, recovery might have worked, but we still re-raise the original exception
                raise e
            except Exception as recovery_error:
                # Recovery failed - update error state
                cuda_error_state["has_cuda_error"] = True
                cuda_error_state["last_error_time"] = time.time()
                cuda_error_state["error_count"] += 1
                cuda_error_state["last_error_message"] = str(e)
                logger.error(f"CUDA error recovery failed: {recovery_error}")
                raise HTTPException(
                    status_code=503, 
                    detail="CUDA error occurred and recovery failed. Please restart the service."
                )
        else:
            # Non-CUDA error, just re-raise
            raise e

@contextmanager
def cuda_error_recovery_no_http():
    """
    Context manager for CUDA error recovery that doesn't raise HTTPException.
    Used inside streaming generators to avoid "response already started" errors.
    """
    global cuda_error_state
    try:
        yield
    except Exception as e:
        # Check if this is a CUDA-related error
        error_str = str(e).lower()
        if 'cuda' in error_str or 'gpu' in error_str or torch.cuda.is_available():
            try:
                # Attempt CUDA recovery
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                logger.info("CUDA recovery attempted (stream)")
                # Mark the error but don't raise HTTPException
                cuda_error_state["has_cuda_error"] = True
                cuda_error_state["last_error_time"] = time.time()
                cuda_error_state["error_count"] += 1
                cuda_error_state["last_error_message"] = str(e)
                logger.error(f"CUDA error in stream: {e}")
            except Exception as recovery_error:
                # Recovery failed - update error state
                cuda_error_state["has_cuda_error"] = True
                cuda_error_state["last_error_time"] = time.time()
                cuda_error_state["error_count"] += 1
                cuda_error_state["last_error_message"] = str(e)
                logger.error(f"CUDA error (stream) and recovery failed: {recovery_error}")
        # Swallow the exception to end stream gracefully
        return

@app.post("/synthesize")
async def synthesize(payload: SynthesizePayload):
    try:
        # Check VRAM headroom before starting
        if not ensure_vram():
            raise HTTPException(
                status_code=503,
                detail="Insufficient GPU memory. Please try again later."
            )
        
        # Record start time for performance measurement
        start_time = time.time()
        
        # Generate audio using the streaming TTS method which accepts diffusion parameters
        try:
            logger.info(f"Starting synthesis for voice '{payload.voice}' with text: '{payload.text[:50]}...'")
            voice_samples = _cached_voice_samples(payload.voice)
            if voice_samples:
                logger.info(f"Loaded {len(voice_samples)} voice samples for '{payload.voice}'")
            else:
                logger.info("Using random voice generation")
        except ValueError as e:
            logger.error(f"Voice loading failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        
        # Validate voice samples before proceeding
        if not voice_samples:
            logger.error(f"No voice samples available for voice '{payload.voice}'")
            raise HTTPException(status_code=400, detail=f"No voice samples available for voice '{payload.voice}'")
        
        # Serialize GPU access with semaphore
        async with gpu_semaphore:
            with cuda_error_recovery():
                audio_generator = tts.tts_stream(
                    payload.text,
                    voice_samples=voice_samples,
                    **get_preset_settings(payload.preset)
                )
                
                # Get the first (and only) audio chunk from the generator
                pcm_audio = next(audio_generator)
        
        # Record generation time
        generation_time = time.time() - start_time
        
        # Convert tensor to numpy array
        if isinstance(pcm_audio, torch.Tensor):
            audio_data = pcm_audio.cpu().numpy()
        else:
            audio_data = pcm_audio
        
        # Handle multi-dimensional audio data
        if len(audio_data.shape) > 1:
            # Take the first channel if multi-channel
            audio_data = audio_data[0] if audio_data.shape[0] == 1 else audio_data.flatten()
        
        # Calculate audio length and RTF
        audio_length = len(audio_data) / 24000  # 24kHz sample rate
        rtf = generation_time / audio_length
        
        # Log performance metrics
        print(f"🎤 Generated {audio_length:.2f}s audio in {generation_time:.2f}s")
        
        # Real-time is defined as generation time < 500ms
        is_realtime = generation_time < 0.5
        rtf_status = "(REAL-TIME!)" if is_realtime else "(slower than real-time)"
        print(f"⚡ RTF: {rtf:.2f}x | Gen Time: {generation_time*1000:.0f}ms {rtf_status}")
        
        # Normalize audio to 16-bit PCM range
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Convert to WAV format using scipy
        bio = io.BytesIO()
        wavfile.write(bio, 24000, audio_data)  # 24000 Hz sample rate (Tortoise default)
        bio.seek(0)
        
        # Add performance headers
        headers = {
            "X-Generation-Time": str(generation_time),
            "X-Generation-Time-Ms": str(int(generation_time * 1000)),
            "X-Audio-Length": str(audio_length),
            "X-RTF": str(rtf),
            "X-Preset": payload.preset,
            "X-Voice": payload.voice,
            "X-Is-Realtime": str(generation_time < 0.5)
        }
        
        return StreamingResponse(bio, media_type="audio/wav", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize_stream")
async def synthesize_stream(payload: SynthesizePayload):
    """
    Streaming synthesis endpoint for real-time audio generation.
    Returns audio chunks as they are generated.
    """
    try:
        # Check VRAM headroom before starting
        if not ensure_vram():
            raise HTTPException(
                status_code=503,
                detail="Insufficient GPU memory. Please try again later."
            )
        
        def generate_audio():
            """Generator function that safely handles CUDA errors without raising HTTPException."""
            try:
                logger.info(f"Starting streaming synthesis for voice '{payload.voice}'")
                voice_samples = _cached_voice_samples(payload.voice)
                if voice_samples:
                    logger.info(f"Loaded {len(voice_samples)} voice samples for streaming")
                else:
                    logger.error(f"Voice loading failed for '{payload.voice}' - no voice samples loaded")
                    logger.error("Streaming synthesis failed due to voice loading error")
                    return
            except ValueError as e:
                logger.error(f"Voice loading failed in streaming: {e}")
                logger.error("Streaming synthesis failed due to voice loading error")
                return
            
            # Validate voice samples before proceeding
            if not voice_samples:
                logger.error(f"No voice samples available for voice '{payload.voice}'")
                return
            
            # Use threading lock for synchronous GPU access in generator
            with gpu_lock:
                with cuda_error_recovery_no_http():
                    try:
                        logger.info(f"text: {payload.text}")
                        audio_generator = tts.tts_stream(
                            payload.text,
                            voice_samples=voice_samples,
                            **get_preset_settings(payload.preset),
                        )
                        
                        for audio_chunk in audio_generator:
                            if isinstance(audio_chunk, torch.Tensor):
                                audio_data = audio_chunk.cpu().numpy()
                            else:
                                audio_data = audio_chunk
                            
                            # Handle multi-dimensional audio data
                            if len(audio_data.shape) > 1:
                                audio_data = audio_data[0] if audio_data.shape[0] == 1 else audio_data.flatten()
                            
                            # Normalize audio to 16-bit PCM range
                            audio_data = np.clip(audio_data, -1.0, 1.0)
                            audio_data = (audio_data * 32767).astype(np.int16)
                            
                            # Convert to WAV format
                            bio = io.BytesIO()
                            wavfile.write(bio, 24000, audio_data)
                            bio.seek(0)
                            yield bio.read()
                    except Exception as e:
                        logger.exception(f"Streaming synthesis failed: {e}")
                        # DO NOT raise; just stop the stream
                        return
        
        return StreamingResponse(generate_audio(), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {
        "status": "healthy",
        "device": device,
        "platform": platform.system(),
        "concurrency_control": {
            "gpu_semaphore_limit": 1,
            "gpu_lock_enabled": True
        },
        "optimizations": {
            "use_deepspeed": use_deepspeed,
            "kv_cache": True,
            "half": True,
            "autoregressive_batch_size": 4,  # Conservative for stability
            "voice_caching": True
        }
    }

@app.get("/health/cuda")
async def cuda_health_check():
    """
    CUDA-specific health check endpoint to detect CUDA error recovery failures.
    Returns 503 status when CUDA errors have occurred and recovery failed.
    This endpoint is designed for container restart detection.
    """
    global cuda_error_state
    
    if cuda_error_state["has_cuda_error"]:
        # Calculate time since last error
        time_since_error = time.time() - cuda_error_state["last_error_time"] if cuda_error_state["last_error_time"] else 0
        
        # Return unhealthy status with error details
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "reason": "CUDA error recovery failed",
                "error_count": cuda_error_state["error_count"],
                "last_error_message": cuda_error_state["last_error_message"],
                "time_since_error_seconds": time_since_error,
                "recommendation": "Container restart required"
            }
        )
    
    # Return healthy status with CUDA info
    cuda_info = {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "error_count": cuda_error_state["error_count"]
    }
    
    if torch.cuda.is_available():
        cuda_info.update({
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_reserved": torch.cuda.memory_reserved(0)
        })
    
    return cuda_info

@app.get("/voices")
async def list_voices():
    """List all available voices."""
    voices_dir = os.path.join("tortoise", "voices")
    try:
        if os.path.exists(voices_dir):
            voices = [d for d in os.listdir(voices_dir) if os.path.isdir(os.path.join(voices_dir, d))]
            return {
                "voices": voices,
                "total": len(voices),
                "note": "Use 'random' for random voice selection"
            }
        else:
            return {"error": "Voices directory not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/clear-voice-cache")
async def clear_voice_cache_endpoint():
    """Clear the voice samples cache. Useful for debugging voice loading issues."""
    try:
        clear_voice_cache()
        return {"message": "Voice cache cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear voice cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/concurrency")
async def concurrency_status():
    """Monitor concurrency control status and GPU utilization."""
    try:
        status = {
            "gpu_semaphore": {
                "available": gpu_semaphore._value,
                "total": 1,
                "waiting": 0  # Would need to track this separately if needed
            },
            "gpu_lock": {
                "locked": gpu_lock.locked(),
                "owner": gpu_lock._owner if hasattr(gpu_lock, '_owner') else None
            },
            "vram_status": {
                "available": ensure_vram(),
                "min_required_gb": 1
            }
        }
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            total = torch.cuda.get_device_properties(0).total_memory
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
            free = total - reserved
            
            status["vram_status"].update({
                "total_gb": round(total / 1e9, 2),
                "reserved_gb": round(reserved / 1e9, 2),
                "allocated_gb": round(allocated / 1e9, 2),
                "free_gb": round(free / 1e9, 2),
                "utilization_percent": round((reserved / total) * 100, 1)
            })
        
        return status
    except Exception as e:
        logger.error(f"Error getting concurrency status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def load_voice_samples(voice_name, max_retries=3):
    """
    Load voice samples for a given voice name with retry logic.
    Returns None for random voice, or a list of audio tensors for specific voices.
    """
    logger.info(f"Loading voice samples for: '{voice_name}'")
    
    if voice_name == "random":
        logger.info("Using random voice (no conditioning)")
        return None
    
    voice_dir = os.path.join("tortoise", "voices", voice_name)
    logger.info(f"Voice directory: {voice_dir}")
    
    if not os.path.exists(voice_dir):
        logger.error(f"Voice directory not found: {voice_dir}")
        raise ValueError(f"Voice '{voice_name}' not found in voices directory")
    
    # List all audio files first
    audio_files = [f for f in os.listdir(voice_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    logger.info(f"Found {len(audio_files)} audio files: {audio_files}")
    
    # Also check for .pth files (pre-computed conditioning latents)
    pth_files = [f for f in os.listdir(voice_dir) if f.endswith('.pth')]
    logger.info(f"Found {len(pth_files)} .pth files: {pth_files}")
    
    voice_samples = []
    failed_files = []
    
    for file in audio_files:
        file_path = os.path.join(voice_dir, file)
        loaded = False
        
        # Retry loading each file up to max_retries times
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading {file} (attempt {attempt + 1}/{max_retries})")
                audio = load_audio(file_path, 22050)  # 22050 Hz for voice samples
                logger.info(f"Loaded {file}: shape={audio.shape}, dtype={audio.dtype}, device={audio.device if hasattr(audio, 'device') else 'unknown'}")
                voice_samples.append(audio)
                logger.info(f"Successfully loaded {file}")
                loaded = True
                break
            except Exception as e:
                logger.warning(f"Failed to load {file} on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying {file}...")
                    time.sleep(0.1)  # Small delay between retries
                continue
        
        if not loaded:
            logger.error(f"Failed to load {file} after {max_retries} attempts")
            failed_files.append(file)
    
    logger.info(f"Voice loading results for '{voice_name}': {len(voice_samples)} loaded, {len(failed_files)} failed")
    
    if failed_files:
        logger.warning(f"Failed to load files: {failed_files}")
    
    if not voice_samples:
        logger.error(f"No valid audio files loaded from: {voice_dir}")
        raise ValueError(f"No valid audio files found in voice directory: {voice_dir}")
    
    logger.info(f"Successfully loaded {len(voice_samples)} voice samples for '{voice_name}'")
    return voice_samples

if __name__ == "__main__":
    import uvicorn
    
    # Operational hardening recommendations
    print("🚀 Production-ready Tortoise TTS API starting...")
    print("📋 Key operational settings:")
    print(f"   • GPU concurrency: Serialized (semaphore limit: 1)")
    print(f"   • Autoregressive batch size: 4 (conservative)")
    print(f"   • Voice caching: Enabled (max 64 voices)")
    print(f"   • VRAM guard: 1GB minimum free")
    print(f"   • DeepSpeed: {'Enabled' if use_deepspeed else 'Disabled'}")
    print("")
    print("💡 Production deployment tips:")
    print("   • Use --workers 1 with uvicorn (one worker per GPU)")
    print("   • Set CUDA_LAUNCH_BLOCKING=1 in staging for debugging")
    print("   • Enable GPU persistence mode: nvidia-smi -pm 1")
    print("   • Monitor /status/concurrency for GPU utilization")
    print("   • Use /health/cuda for automated restart detection")
    print("")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        # Production recommendations:
        # workers=1,  # Uncomment for production - one worker per GPU
        # access_log=True,  # Enable access logging
        # log_level="info"
    )
