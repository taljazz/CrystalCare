import numpy as np
import sounddevice as sd
import wave
import os
from typing import Optional, Callable, List
import logging
import threading
import gc
import wx

class SoundPlayer:
    """
    Manages sound playback and saving to WAV files, ensuring GUI responsiveness via threading.
    """
    def __init__(self, sound_generator) -> None:
        self.sound_generator = sound_generator

    def play_sound(self, duration: float, base_freq: float, sample_rate: int = 48000,
                   stop_event: Optional[threading.Event] = None,
                   update_status: Optional[Callable[[str], None]] = None,
                   on_complete: Optional[Callable[[], None]] = None,
                   dimensional_mode: bool = False,
                   **kwargs) -> None:
        """
        Play sound with 50ms poll-based stream monitoring.
        Uses threading.Event for efficient stop signaling.
        """
        try:
            progress_gauge = kwargs.get('progress_gauge')
            update_progress = None
            if progress_gauge:
                update_progress = lambda p: wx.CallAfter(progress_gauge.SetValue, int(p * 100))

            if update_status:
                update_status("Generating sound...")
            sound_data = self.sound_generator.generate_dynamic_sound(duration, base_freq, sample_rate,
                                                                     interval_duration_list=[30, 45, 60, 75, 90],
                                                                     stop_event=stop_event,
                                                                     update_progress=update_progress,
                                                                     dimensional_mode=dimensional_mode)
            if stop_event and stop_event.is_set():
                logging.debug("Playback aborted due to stop_event.")
                if update_status:
                    update_status("Playback stopped.")
                return
            if update_status:
                update_status("Playing sound...")

            # Poll-based playback monitoring with 50ms responsiveness
            playback_finished = threading.Event()
            was_stopped = False

            sd.play(sound_data, samplerate=sample_rate, blocking=False)

            # Wait for either: playback completion OR stop_event
            while not playback_finished.is_set():
                # Check stop_event with short timeout for responsive cancellation
                if stop_event and stop_event.wait(timeout=0.05):  # 50ms responsiveness
                    sd.stop()
                    was_stopped = True
                    if update_status:
                        update_status("Playback stopped manually.")
                    break

                # Check if playback finished naturally
                stream = sd.get_stream()
                if stream is None or not stream.active:
                    playback_finished.set()
                    break

            # Free sound data after playback
            del sound_data
            gc.collect()

            if not was_stopped and update_status:
                update_status("Playback finished.")
        except sd.PortAudioError as e:
            logging.error(f"Playback failed: {e}")
            if update_status:
                update_status(f"Playback error: {e}")
        except Exception as e:
            logging.exception(f"Unexpected playback error: {e}")
            if update_status:
                update_status(f"Playback error: {e}")
        finally:
            if on_complete:
                on_complete()

    def save_to_wav(self, duration: float, base_freq: float, filename: str, sample_rate: int = 48000,
                    stop_event: Optional[threading.Event] = None,
                    update_status: Optional[Callable[[str], None]] = None,
                    on_complete: Optional[Callable[[], None]] = None,
                    dimensional_mode: bool = False,
                    **kwargs) -> None:
        try:
            progress_gauge = kwargs.get('progress_gauge')
            update_progress = None
            if progress_gauge:
                update_progress = lambda p: wx.CallAfter(progress_gauge.SetValue, int(p * 100))
            
            if update_status:
                update_status("Generating sound for saving...")
            sound_data = self.sound_generator.generate_dynamic_sound(duration, base_freq, sample_rate,
                                                                     interval_duration_list=[30, 45, 60, 75, 90],
                                                                     stop_event=stop_event,
                                                                     update_progress=update_progress,
                                                                     dimensional_mode=dimensional_mode)
            if stop_event and stop_event.is_set():
                logging.debug("WAV generation aborted due to stop_event.")
                if update_status:
                    update_status("Saving stopped.")
                return
            # Single max calculation for efficiency
            max_val = np.abs(sound_data).max()
            if max_val == 0:
                if update_status:
                    update_status("Sound generation was stopped. Nothing to save.")
                return
            scale_factor = np.float32(32767.0 / max_val)
            if update_status:
                update_status("Saving to WAV file...")

            # Chunked WAV writing: only one chunk's worth of int16 data at a time
            n_channels = sound_data.shape[1] if sound_data.ndim == 2 else 1
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(n_channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)

                chunk_size = sample_rate * 10  # 10 seconds at a time
                for start in range(0, len(sound_data), chunk_size):
                    chunk = sound_data[start:start + chunk_size]
                    scaled = np.int16(chunk * scale_factor)
                    wf.writeframes(scaled.tobytes())
                    del scaled

            del sound_data
            gc.collect()

            if not (stop_event and stop_event.is_set()) and update_status:
                update_status(f"Resonance saved as {filename}.")
        except IOError as e:
            logging.error(f"File saving failed: {e}")
            if update_status:
                update_status(f"Error saving file: {e}")
        except Exception as e:
            logging.exception(f"Unexpected error during saving: {e}")
            if update_status:
                update_status(f"Error: {e}")
        finally:
            if on_complete:
                on_complete()

    def batch_save(self, duration: float, base_freqs: List[float], save_dir: str, num_tones: int,
                   sample_rate: int = 48000,
                   stop_event: Optional[threading.Event] = None,
                   update_status: Optional[Callable[[str], None]] = None,
                   on_complete: Optional[Callable[[], None]] = None,
                   dimensional_mode: bool = False,
                   **kwargs) -> None:
        """
        Sequential batch save - generates and saves tones one at a time.
        This avoids resource contention and provides predictable memory usage.
        Sacred layers within each tone still run in parallel (3 workers).
        """
        try:
            progress_gauge = kwargs.get('progress_gauge')
            completed_count = 0

            def generate_and_save_tone(tone_index: int) -> Optional[str]:
                """Generate and save a single tone. Returns filename on success, None on failure/cancel."""
                if stop_event and stop_event.is_set():
                    return None

                filename = os.path.join(save_dir, f"Tone{tone_index + 1}.wav")
                base_freq = base_freqs[tone_index]

                try:
                    # Generate sound data
                    sound_data = self.sound_generator.generate_dynamic_sound(
                        duration, base_freq, sample_rate,
                        interval_duration_list=[30, 45, 60, 75, 90],
                        stop_event=stop_event,
                        update_progress=None,
                        dimensional_mode=dimensional_mode
                    )

                    if stop_event and stop_event.is_set():
                        return None

                    # Scale and save with chunked WAV writing
                    max_val = np.max(np.abs(sound_data))
                    if max_val == 0:
                        logging.warning(f"Tone {tone_index + 1} generated silence, skipping.")
                        del sound_data
                        return None
                    scale_factor = np.float32(32767.0 / max_val)

                    n_channels = sound_data.shape[1] if sound_data.ndim == 2 else 1
                    with wave.open(filename, 'wb') as wf:
                        wf.setnchannels(n_channels)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(sample_rate)

                        chunk_size = sample_rate * 10  # 10 seconds at a time
                        for start in range(0, len(sound_data), chunk_size):
                            chunk = sound_data[start:start + chunk_size]
                            scaled = np.int16(chunk * scale_factor)
                            wf.writeframes(scaled.tobytes())
                            del scaled

                    del sound_data
                    gc.collect()

                    return filename
                except Exception as e:
                    logging.exception(f"Error generating tone {tone_index + 1}")
                    return None

            if update_status:
                update_status("Starting sequential batch save...")

            # Reset progress gauge for overall batch progress
            if progress_gauge:
                wx.CallAfter(progress_gauge.SetValue, 0)

            # Process tones sequentially - one at a time for optimal memory usage
            for i in range(num_tones):
                if stop_event and stop_event.is_set():
                    if update_status:
                        update_status("Batch save stopped by user.")
                    break

                if update_status:
                    update_status(f"Generating tone {i + 1}/{num_tones}...")

                result = generate_and_save_tone(i)

                if result:
                    completed_count += 1
                    progress = completed_count / num_tones
                    if update_status:
                        update_status(f"Saved {result} ({completed_count}/{num_tones})")
                    if progress_gauge:
                        wx.CallAfter(progress_gauge.SetValue, int(progress * 100))
                else:
                    if update_status and not (stop_event and stop_event.is_set()):
                        update_status(f"Error saving tone {i + 1}")

            if not (stop_event and stop_event.is_set()) and update_status:
                update_status(f"Batch save completed. {completed_count}/{num_tones} tones saved.")
        except Exception as e:
            logging.exception("Error during batch save.")
            if update_status:
                update_status(f"Batch save error: {e}")
        finally:
            if on_complete:
                on_complete()

    def stop_playback(self) -> None:
        try:
            sd.stop()
            logging.debug("Playback stopped successfully.")
        except Exception as e:
            logging.error(f"Error stopping playback: {e}")