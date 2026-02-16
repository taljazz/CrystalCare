import wx
import webbrowser
import threading
import numpy as np
import sys
import logging
from frequencies import FrequencyManager
from SoundGenerator import AudioProcessor, SoundGenerator
from SoundManager import SoundPlayer
from typing import Optional, Callable

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class UserGuideManager:
    """
    Manages the opening of the user guide in the default web browser.
    """
    def open_guide(self, update_status: Optional[Callable[[str], None]] = None) -> None:
        try:
            webbrowser.open("Guide.html")
            if update_status:
                update_status("Opened user guide successfully.")
        except Exception as e:
            logging.exception(f"Failed to open user guide: {e}")
            if update_status:
                update_status("Error: Unable to open the user guide.")

class NumericValidator(wx.Validator):
    """
    A wx.Validator that ensures the input is a valid positive number.
    """
    def __init__(self, callback: Optional[Callable[[str], None]] = None) -> None:
        super(NumericValidator, self).__init__()
        self.callback = callback

    def Clone(self) -> 'NumericValidator':
        return NumericValidator(self.callback)

    def Validate(self, parent) -> bool:
        text_ctrl = self.GetWindow()
        text = text_ctrl.GetValue().strip()
        if not text:
            text_ctrl.SetBackgroundColour(wx.Colour(255, 192, 192))
            text_ctrl.Refresh()
            if self.callback:
                self.callback("Please enter a value.")
            return False
        try:
            value = float(text)
            if value <= 0:
                raise ValueError("Duration must be positive.")
            text_ctrl.SetBackgroundColour(wx.Colour(255, 255, 255))
            text_ctrl.Refresh()
            if self.callback:
                self.callback("Input valid.")
            return True
        except ValueError as e:
            text_ctrl.SetBackgroundColour(wx.Colour(255, 192, 192))
            text_ctrl.Refresh()
            if self.callback:
                self.callback(f"Error: {str(e)}")
            return False

    def TransferToWindow(self) -> bool:
        return True

    def TransferFromWindow(self) -> bool:
        return True

class CrystalCareFrame(wx.Frame):
    """
    The main GUI frame for the CrystalCare application.
    """
    # Thread join timeout in seconds
    THREAD_JOIN_TIMEOUT = 5.0

    FREQUENCY_MODES = {
        0: "Standard",
        1: "Solfeggio",
        2: "Fibonacci",
        3: "Pythagorean",
        4: "Triple Helix DNA Activation",
        5: "Taygetan Binaural",
        6: "Dimensional Shift",
    }

    def __init__(self, parent, title: str,
                 frequency_manager: FrequencyManager,
                 audio_processor: AudioProcessor,
                 sound_player: SoundPlayer,
                 user_guide_manager: UserGuideManager) -> None:
        super(CrystalCareFrame, self).__init__(parent, title=title, size=(600, 600))
        self.frequency_manager = frequency_manager
        self.audio_processor = audio_processor
        self.sound_player = sound_player
        self.sound_generator = sound_player.sound_generator
        self.user_guide_manager = user_guide_manager
        self.init_ui()
        self.Centre()
        self.Show()
        self.stop_event = threading.Event()
        self.current_thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()  # Protect thread reference access
        self._thread_timer: Optional[wx.Timer] = None  # Timer for non-blocking thread completion
        self._is_closing = False  # Guard against wx.CallAfter on destroyed window

        # Bind close event for graceful shutdown
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def init_ui(self) -> None:
        self.panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        freq_box = wx.StaticBox(self.panel, label="Select Frequency Set")
        freq_sizer = wx.StaticBoxSizer(freq_box, wx.VERTICAL)
        self.freq_choice = wx.Choice(self.panel, choices=[
            "1. Lower Frequencies (174Hz, 396Hz, 417Hz, 528Hz)",
            "2. Higher Frequencies (852Hz, 963Hz)",
            "3. Atlantean Cosmic Frequencies (136.10Hz, 194.18Hz, 211.44Hz, 303Hz)",
            "4. Combined Mode (Sacred Geometry + Flower of Life)",
            "5. Triple Helix DNA Activation Mode",
            "6. Taygetan Resonances (DNA Activation with Binaural Sync)",
            "7. Dimensional Journey (1D-9D Realignment)"
        ])
        self.freq_choice.SetSelection(0)
        self.freq_choice.SetToolTip("Choose a set of frequencies for your session.")
        freq_sizer.Add(self.freq_choice, flag=wx.ALL | wx.EXPAND, border=10)
        vbox.Add(freq_sizer, flag=wx.ALL | wx.EXPAND, border=10)
        duration_box = wx.StaticBox(self.panel, label="Session Duration (Minutes)")
        duration_sizer = wx.StaticBoxSizer(duration_box, wx.HORIZONTAL)
        self.duration_text = wx.TextCtrl(self.panel, validator=NumericValidator(callback=self.on_validation))
        self.duration_text.SetToolTip("Enter the duration of the session in minutes (max 60).")
        duration_sizer.Add(self.duration_text, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)
        vbox.Add(duration_sizer, flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=10)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.play_btn = wx.Button(self.panel, label="Play")
        self.play_btn.SetToolTip("Generate and play the sound session.")
        self.save_btn = wx.Button(self.panel, label="Save as WAV")
        self.save_btn.SetToolTip("Generate and save the sound session as a WAV file.")
        self.guide_btn = wx.Button(self.panel, label="Open Guide")
        self.guide_btn.SetToolTip("Open the CrystalCare user guide for assistance.")
        self.stop_btn = wx.Button(self.panel, label="Stop")
        self.stop_btn.SetToolTip("Stop the ongoing operation.")
        self.stop_btn.Hide()
        self.batch_save_btn = wx.Button(self.panel, label="Batch Save")
        self.batch_save_btn.SetToolTip("Generate and save multiple tones in a selected folder.")
        btn_sizer.Add(self.play_btn, flag=wx.ALL, border=5)
        btn_sizer.Add(self.save_btn, flag=wx.ALL, border=5)
        btn_sizer.Add(self.guide_btn, flag=wx.ALL, border=5)
        btn_sizer.Add(self.batch_save_btn, flag=wx.ALL, border=5)
        btn_sizer.Add(self.stop_btn, flag=wx.ALL, border=5)
        vbox.Add(btn_sizer, flag=wx.ALIGN_CENTER)
        self.status_text = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        vbox.Add(self.status_text, proportion=1, flag=wx.ALL | wx.EXPAND, border=10)
        self.gauge = wx.Gauge(self.panel, range=100, style=wx.GA_HORIZONTAL | wx.GA_SMOOTH)
        self.gauge.SetToolTip("Progress of sound generation")
        vbox.Add(self.gauge, 0, flag=wx.ALL | wx.EXPAND, border=10)
        self.gauge.Hide()
        self.panel.SetSizer(vbox)
        self.play_btn.Bind(wx.EVT_BUTTON, self.on_play)
        self.save_btn.Bind(wx.EVT_BUTTON, self.on_save)
        self.guide_btn.Bind(wx.EVT_BUTTON, self.on_open_guide)
        self.stop_btn.Bind(wx.EVT_BUTTON, self.on_stop)
        self.batch_save_btn.Bind(wx.EVT_BUTTON, self.on_batch_save)

    def on_validation(self, message: str) -> None:
        self.update_status(message)

    def on_play(self, event) -> None:
        with self._thread_lock:
            if self.current_thread is not None and self.current_thread.is_alive():
                self.update_status("An operation is already in progress.")
                return
        try:
            freq_selection = self.freq_choice.GetSelection()
            frequencies = self.frequency_manager.get_frequencies(freq_selection)
            if not frequencies and freq_selection != 6:
                self.update_status("Error: No frequencies available for the selected set.")
                return
            duration_str = self.duration_text.GetValue().strip()
            if not self.validate_input(duration_str):
                return
            duration_minutes = float(duration_str)
            if duration_minutes > 60:
                self.update_status("Error: Duration must be <= 60 minutes.")
                self.duration_text.SetBackgroundColour(wx.Colour(255, 192, 192))
                self.duration_text.Refresh()
                return
            duration_seconds = duration_minutes * 60
            if freq_selection == 6:
                base_freq = 432  # Default for dimensional
            else:
                base_freq = np.random.choice(frequencies) if frequencies and not isinstance(frequencies[0], tuple) else np.random.choice([pair[0] for pair in frequencies]) if frequencies else 432  # Handle binaural
            logging.debug(f"Selected base frequency for Play: {base_freq}")
            self.update_status("Resonating...")
            self.toggle_controls(show_stop=True)
            self.stop_event.clear()
            def on_complete() -> None:
                if self._is_closing:
                    return
                wx.CallAfter(self.toggle_controls, show_stop=False)
                mode_name = self.FREQUENCY_MODES.get(freq_selection, "")
                if freq_selection in (4, 5, 6):
                    msg = f"{mode_name} completed."
                else:
                    msg = "Playback completed."
                wx.CallAfter(self.update_status, msg)
            # Thread-safe thread creation and tracking
            with self._thread_lock:
                self.current_thread = threading.Thread(
                    target=self.sound_player.play_sound,
                    args=(duration_seconds, base_freq),
                    kwargs={
                        'sample_rate': 48000,
                        'stop_event': self.stop_event,
                        'update_status': self.update_status,
                        'on_complete': on_complete,
                        'progress_gauge': self.gauge,
                        'dimensional_mode': freq_selection == 6
                    },
                    daemon=True,  # Keep daemon for cleanup on unexpected exit
                    name="CrystalCare-PlaySound"
                )
                self.current_thread.start()
        except Exception as e:
            logging.exception("Error in on_play.")
            self.update_status(f"Error: {e}")

    def on_save(self, event) -> None:
        with self._thread_lock:
            if self.current_thread is not None and self.current_thread.is_alive():
                self.update_status("An operation is already in progress.")
                return
        try:
            freq_selection = self.freq_choice.GetSelection()
            frequencies = self.frequency_manager.get_frequencies(freq_selection)
            if not frequencies and freq_selection != 6:
                self.update_status("Error: No frequencies available for the selected set.")
                return
            duration_str = self.duration_text.GetValue().strip()
            if not self.validate_input(duration_str):
                return
            duration_minutes = float(duration_str)
            if duration_minutes > 60:
                self.update_status("Error: Duration must be <= 60 minutes.")
                self.duration_text.SetBackgroundColour(wx.Colour(255, 192, 192))
                self.duration_text.Refresh()
                return
            duration_seconds = duration_minutes * 60
            dlg = wx.FileDialog(self, "Save Resonance as WAV", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT, wildcard="WAV files (*.wav)|*.wav")
            if dlg.ShowModal() == wx.ID_OK:
                filename = dlg.GetPath()
                dlg.Destroy()
                if freq_selection == 6:
                    base_freq = 432  # Default for dimensional
                else:
                    base_freq = np.random.choice(frequencies) if frequencies and not isinstance(frequencies[0], tuple) else np.random.choice([pair[0] for pair in frequencies]) if frequencies else 432  # Handle binaural
                logging.debug(f"Selected base frequency for Save: {base_freq}")
                self.update_status("Resonating and saving to file...")
                self.toggle_controls(show_stop=True)
                self.stop_event.clear()
                def on_complete() -> None:
                    if self._is_closing:
                        return
                    wx.CallAfter(self.toggle_controls, show_stop=False)
                    mode_name = self.FREQUENCY_MODES.get(freq_selection, "")
                    if freq_selection in (4, 5, 6):
                        msg = f"{mode_name} saved as {filename}."
                    else:
                        msg = f"Resonance saved as {filename}."
                    wx.CallAfter(self.update_status, msg)
                # Thread-safe thread creation and tracking
                with self._thread_lock:
                    self.current_thread = threading.Thread(
                        target=self.sound_player.save_to_wav,
                        args=(duration_seconds, base_freq, filename),
                        kwargs={
                            'sample_rate': 48000,
                            'stop_event': self.stop_event,
                            'update_status': self.update_status,
                            'on_complete': on_complete,
                            'progress_gauge': self.gauge,
                            'dimensional_mode': freq_selection == 6
                        },
                        daemon=True,  # Keep daemon for cleanup on unexpected exit
                        name="CrystalCare-SaveWAV"
                    )
                    self.current_thread.start()
            else:
                logging.debug("Save dialog cancelled by user.")
                self.update_status("Save operation cancelled by user.")
        except Exception as e:
            logging.exception("Error in on_save.")
            self.update_status(f"Error: {e}")

    def on_stop(self, event) -> None:
        """Stop current operation without blocking the GUI."""
        try:
            self.stop_event.set()
            self.sound_player.stop_playback()
            self.update_status("Stopping operation...")

            # Check if thread needs cleanup - use non-blocking timer approach
            with self._thread_lock:
                thread = self.current_thread
            if thread is not None and thread.is_alive():
                # Start a timer to check thread completion without blocking GUI
                self._start_thread_completion_timer(
                    on_complete=lambda: self._finish_stop(),
                    on_timeout=lambda: self._finish_stop(timed_out=True)
                )
            else:
                self._finish_stop()
        except Exception as e:
            logging.exception("Error stopping playback.")
            self.update_status(f"Error stopping playback: {e}")
            self.toggle_controls(show_stop=False)

    def _finish_stop(self, timed_out: bool = False) -> None:
        """Complete the stop operation after thread finishes."""
        if timed_out:
            logging.warning("Thread did not terminate within timeout")
            self.update_status("Warning: Operation may still be finishing in background...")
        else:
            logging.debug("Thread terminated successfully")
        self.toggle_controls(show_stop=False)
        self.update_status("Operation has been stopped.")

    def _start_thread_completion_timer(self, on_complete: Callable, on_timeout: Callable,
                                        check_interval_ms: int = 100, max_checks: int = 50) -> None:
        """
        Non-blocking timer to wait for thread completion.
        Checks every check_interval_ms (default 100ms) up to max_checks times (default 5 seconds total).
        """
        # Stop any existing timer to prevent race conditions on rapid stop/start
        if self._thread_timer is not None:
            if self._thread_timer.IsRunning():
                self._thread_timer.Stop()
            self._thread_timer.Destroy()
            self._thread_timer = None

        self._thread_check_count = 0
        self._thread_max_checks = max_checks
        self._thread_on_complete = on_complete
        self._thread_on_timeout = on_timeout

        # Create and start timer
        self._thread_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_thread_check_timer, self._thread_timer)
        self._thread_timer.Start(check_interval_ms)

    def _on_thread_check_timer(self, event) -> None:
        """Timer callback to check if thread has finished."""
        self._thread_check_count += 1

        with self._thread_lock:
            thread = self.current_thread

        # Check if thread finished
        if thread is None or not thread.is_alive():
            self._thread_timer.Stop()
            self._thread_on_complete()
            return

        # Check if timed out
        if self._thread_check_count >= self._thread_max_checks:
            self._thread_timer.Stop()
            self._thread_on_timeout()
            return

        # Otherwise, timer will fire again automatically

    def on_close(self, event) -> None:
        """Graceful shutdown - stop any running operations before closing (non-blocking)."""
        logging.debug("Application closing - initiating graceful shutdown")
        self._is_closing = True

        # Signal stop to any running operation
        self.stop_event.set()
        self.sound_player.stop_playback()

        # Check if thread needs cleanup
        with self._thread_lock:
            thread = self.current_thread

        if thread is not None and thread.is_alive():
            self.update_status("Waiting for operation to finish...")
            # Start timer to check thread completion, then close
            self._closing = True
            self._start_thread_completion_timer(
                on_complete=lambda: self._finish_close(),
                on_timeout=lambda: self._finish_close(force=True)
            )
        else:
            self._finish_close()

    def _finish_close(self, force: bool = False) -> None:
        """Complete the close operation after thread finishes."""
        if force:
            logging.warning("Thread did not terminate during shutdown - closing anyway")
        else:
            logging.debug("Graceful shutdown complete")
        # Stop timer before destroying window to prevent stale timer events
        if hasattr(self, '_thread_timer') and self._thread_timer is not None and self._thread_timer.IsRunning():
            self._thread_timer.Stop()
        # Shut down shared sacred layer thread pool
        self.sound_generator.shutdown()
        self.Destroy()  # Close the window

    def toggle_controls(self, show_stop: bool) -> None:
        if show_stop:
            self.freq_choice.Hide()
            self.duration_text.Hide()
            self.play_btn.Hide()
            self.save_btn.Hide()
            self.guide_btn.Hide()
            self.batch_save_btn.Hide()
            self.stop_btn.Show()
            self.gauge.SetValue(0)  # Reset before showing
            self.gauge.Show()
        else:
            self.freq_choice.Show()
            self.duration_text.Show()
            self.play_btn.Show()
            self.save_btn.Show()
            self.guide_btn.Show()
            self.batch_save_btn.Show()
            self.stop_btn.Hide()
            self.gauge.Hide()
            self.gauge.SetValue(0)
        # Force layout update and refresh on both panel and frame
        self.panel.Layout()
        self.Layout()
        self.Refresh()

    def on_open_guide(self, event) -> None:
        self.user_guide_manager.open_guide(update_status=self.update_status)

    def update_status(self, message: str) -> None:
        if self._is_closing:
            return
        if wx.IsMainThread():
            self.status_text.AppendText(message + "\n")
        else:
            wx.CallAfter(self.status_text.AppendText, message + "\n")

    def validate_input(self, input_str: str) -> bool:
        try:
            value = float(input_str)
            if value <= 0:
                self.update_status("Error: Duration must be positive.")
                self.duration_text.SetBackgroundColour(wx.Colour(255, 192, 192))
                self.duration_text.Refresh()
                return False
            else:
                self.duration_text.SetBackgroundColour(wx.Colour(255, 255, 255))
                self.duration_text.Refresh()
                return True
        except ValueError:
            self.update_status("Error: Please enter a numerical value for duration.")
            self.duration_text.SetBackgroundColour(wx.Colour(255, 192, 192))
            self.duration_text.Refresh()
            return False

    def on_batch_save(self, event) -> None:
        with self._thread_lock:
            if self.current_thread is not None and self.current_thread.is_alive():
                self.update_status("An operation is already in progress.")
                return
        try:
            # Validate duration first before showing dialogs
            duration_str = self.duration_text.GetValue().strip()
            if not self.validate_input(duration_str):
                return
            duration_minutes = float(duration_str)
            if duration_minutes > 60:
                self.update_status("Error: Duration must be <= 60 minutes.")
                self.duration_text.SetBackgroundColour(wx.Colour(255, 192, 192))
                self.duration_text.Refresh()
                return
            duration_seconds = duration_minutes * 60

            freq_selection = self.freq_choice.GetSelection()
            frequencies = self.frequency_manager.get_frequencies(freq_selection)
            if not frequencies and freq_selection != 6:
                self.update_status("Error: No frequencies available for the selected set.")
                return

            dlg = wx.NumberEntryDialog(self, "Enter the number of tones to save:", "Batch Save", "Number of Tones", 1, 1, 1000)
            if dlg.ShowModal() == wx.ID_OK:
                num_tones = dlg.GetValue()
                dlg.Destroy()
            else:
                dlg.Destroy()
                self.update_status("Batch save cancelled by user.")
                return
            dir_dlg = wx.DirDialog(self, "Choose a directory to save the tones:", style=wx.DD_DEFAULT_STYLE)
            if dir_dlg.ShowModal() == wx.ID_OK:
                save_dir = dir_dlg.GetPath()
                dir_dlg.Destroy()
            else:
                dir_dlg.Destroy()
                self.update_status("Batch save cancelled by user.")
                return

            if freq_selection == 6:
                base_freqs = [432] * num_tones  # Default for dimensional
            else:
                base_freqs = [np.random.choice(frequencies) if frequencies and not isinstance(frequencies[0], tuple) else np.random.choice([pair[0] for pair in frequencies]) if frequencies else [432] for _ in range(num_tones)]
            logging.debug(f"Selected base frequencies for Batch Save: {base_freqs}")
            self.update_status(f"Starting batch save of {num_tones} tones to {save_dir}...")
            self.toggle_controls(show_stop=True)
            self.stop_event.clear()
            def on_complete() -> None:
                if self._is_closing:
                    return
                wx.CallAfter(self.toggle_controls, show_stop=False)
                wx.CallAfter(self.update_status, "Batch save completed.")
            # Thread-safe thread creation and tracking
            with self._thread_lock:
                self.current_thread = threading.Thread(
                    target=self.sound_player.batch_save,
                    args=(duration_seconds, base_freqs, save_dir, num_tones),
                    kwargs={
                        'sample_rate': 48000,
                        'stop_event': self.stop_event,
                        'update_status': self.update_status,
                        'on_complete': on_complete,
                        'progress_gauge': self.gauge,
                        'dimensional_mode': freq_selection == 6
                    },
                    daemon=True,  # Keep daemon for cleanup on unexpected exit
                    name="CrystalCare-BatchSave"
                )
                self.current_thread.start()
        except Exception as e:
            logging.exception("Error in on_batch_save.")
            self.update_status(f"Error: {e}")

class App:
    """
    The application entry point that initializes all components and starts the GUI.
    """
    def __init__(self) -> None:
        self.frequency_manager = FrequencyManager()
        self.audio_processor = AudioProcessor()
        self.sound_generator = SoundGenerator(self.frequency_manager, self.audio_processor)
        self.sound_player = SoundPlayer(self.sound_generator)
        self.user_guide_manager = UserGuideManager()

    def run(self) -> None:
        app = wx.App(False)
        CrystalCareFrame(None, "CrystalCare - Sound Healing",
                         frequency_manager=self.frequency_manager,
                         audio_processor=self.audio_processor,
                         sound_player=self.sound_player,
                         user_guide_manager=self.user_guide_manager)
        app.MainLoop()

if __name__ == "__main__":
    try:
        application = App()
        application.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user. Exiting...")
    except Exception as e:
        print(f"An unexpected error occurred in the main application: {e}")