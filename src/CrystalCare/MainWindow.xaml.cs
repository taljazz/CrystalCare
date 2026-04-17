using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Automation;
using System.Windows.Automation.Peers;
using CrystalCare.Audio;
using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Generation;
using Microsoft.Win32;
using NAudio.Wave;

namespace CrystalCare;

/// <summary>
/// Main application window — WPF GUI with full screen reader accessibility.
/// Handles all user interaction: Play, Save, Batch Save, Stop, and Guide.
/// Uses WindowsFormsHost for a native Win32 TextBox that NVDA can navigate reliably.
/// </summary>
public partial class MainWindow : Window
{
    // Core services and state for the application's audio pipeline and UI lifecycle.
    #region Fields

    // Manages all sacred geometry ratio sets and frequency mode lookups
    private readonly FrequencyManager _frequencyManager = new();

    // Orchestrates the 16-stage audio pipeline (streaming and batch generation)
    private readonly SoundGenerator _soundGenerator;

    // Handles NAudio playback (streaming) and WAV file writing (saves)
    private readonly SoundPlayer _soundPlayer;

    // Cancellation source for stopping playback/save operations mid-stream
    private CancellationTokenSource? _cts;

    // Guards against UI updates after the window has begun closing
    private bool _isClosing;

    // Native Win32 multiline TextBox hosted via WindowsFormsHost — screen readers
    // (NVDA, JAWS) can navigate this with arrow keys, unlike WPF's LiveRegion
    private readonly System.Windows.Forms.TextBox _statusTextBox;

    #endregion

    // Initializes the WPF window, creates the accessible status TextBox,
    // wires up the sound generator and player, and posts the ready message.
    #region Constructor

    public MainWindow()
    {
        InitializeComponent();

        // Create a native Win32 multiline read-only TextBox for screen reader access.
        // WPF's AutomationPeer LiveRegionChanged is broken in NVDA, so we use
        // WindowsFormsHost with a standard WinForms TextBox instead.
        _statusTextBox = new System.Windows.Forms.TextBox
        {
            Multiline = true,
            ReadOnly = true,
            ScrollBars = System.Windows.Forms.ScrollBars.Vertical,
            WordWrap = true,
            Dock = System.Windows.Forms.DockStyle.Fill,
            Font = new System.Drawing.Font("Segoe UI", 9f),
            AccessibleName = "Status messages",
            AccessibleRole = System.Windows.Forms.AccessibleRole.Text,
        };
        StatusHost.Child = _statusTextBox;

        // Create the sound generator with the frequency manager for ratio lookups
        _soundGenerator = new SoundGenerator(_frequencyManager);

        // Create the sound player that wraps NAudio for playback and saving
        _soundPlayer = new SoundPlayer(_soundGenerator);

        // Populate the output device dropdown with all available audio devices
        PopulateDeviceList();

        UpdateStatus("CrystalCare ready. Select a frequency set and duration, then press Play.");
    }

    #endregion

    // Button click handlers for Play, Save, Guide, Batch Save, Stop, and window close.
    // Each handler validates input, derives the base frequency for the selected mode,
    // and delegates to the SoundPlayer for streaming playback or WAV file generation.
    #region Event Handlers

    /// <summary>
    /// Play button — starts real-time streaming playback through speakers.
    /// Audio begins within ~0.2 seconds via the producer-consumer pattern.
    /// </summary>
    private async void OnPlay_Click(object sender, RoutedEventArgs e)
    {
        if (!ValidateAndParseDuration(out float duration)) return;

        // Cast the ComboBox index directly to FrequencyMode enum (values match 0-6)
        var mode = (FrequencyMode)FreqChoice.SelectedIndex;
        float baseFreq = DeriveBaseFreq(mode);

        _cts = new CancellationTokenSource();
        ToggleControls(isPlaying: true);
        UpdateStatus("Resonating...");

        try
        {
            // Stream audio in real-time — constant memory regardless of duration.
            // Pass the selected output device number for playback.
            await _soundPlayer.PlayStreamAsync(
                duration * 60f, baseFreq, 48000, _cts.Token,
                updateStatus: UpdateStatus,
                freqMode: mode,
                deviceNumber: GetSelectedDeviceNumber());
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            UpdateStatus($"Error: {ex.Message}");
        }
        finally
        {
            if (!_isClosing)
            {
                ToggleControls(isPlaying: false);

                // Named modes get personalized completion messages
                string msg = mode is FrequencyMode.TripleHelixDna or FrequencyMode.TaygetanBinaural or FrequencyMode.DimensionalShift
                    ? $"{GetModeName(mode)} completed."
                    : "Playback completed.";
                UpdateStatus(msg);
            }
        }
    }

    /// <summary>
    /// Save button — generates the full session in memory and writes a WAV file.
    /// Two-pass: generate audio, find peak, write normalized 16-bit PCM.
    /// </summary>
    private async void OnSave_Click(object sender, RoutedEventArgs e)
    {
        if (!ValidateAndParseDuration(out float duration)) return;

        var mode = (FrequencyMode)FreqChoice.SelectedIndex;

        // Guard: ensure frequencies exist for the selected mode (Dimensional uses all)
        if (mode != FrequencyMode.DimensionalShift)
        {
            var freqResult = _frequencyManager.GetFrequencies(mode);
            if (freqResult.Frequencies.Length == 0 && !freqResult.IsBinaural)
            {
                UpdateStatus("Error: No frequencies available for the selected set.");
                return;
            }
        }

        // Show save dialog for WAV file location
        var dialog = new Microsoft.Win32.SaveFileDialog
        {
            Title = "Save Resonance as WAV",
            Filter = "WAV files (*.wav)|*.wav",
            DefaultExt = ".wav",
            FileName = "CrystalCare_session.wav",
            OverwritePrompt = true,
        };

        if (dialog.ShowDialog() != true)
        {
            UpdateStatus("Save operation cancelled by user.");
            return;
        }

        string filename = dialog.FileName;
        float baseFreq = DeriveBaseFreq(mode);

        _cts = new CancellationTokenSource();
        ToggleControls(isPlaying: true, showGauge: true);
        UpdateStatus("Resonating and saving to file...");

        try
        {
            // Generate full session in memory and write to WAV
            await _soundPlayer.SaveToWavAsync(
                filename, duration * 60f, baseFreq, 48000, _cts.Token,
                updateStatus: UpdateStatus,
                updateProgress: p => Dispatcher.InvokeAsync(() => Gauge.Value = p * 100),
                freqMode: mode);
        }
        catch (OperationCanceledException)
        {
            UpdateStatus("Save cancelled.");
        }
        catch (Exception ex)
        {
            UpdateStatus($"Save error: {ex.Message}");
        }
        finally
        {
            if (!_isClosing)
            {
                ToggleControls(isPlaying: false, showGauge: false);
                if (!(_cts?.IsCancellationRequested ?? true))
                {
                    string msg = mode is FrequencyMode.TripleHelixDna or FrequencyMode.TaygetanBinaural or FrequencyMode.DimensionalShift
                        ? $"{GetModeName(mode)} saved as {filename}."
                        : $"Resonance saved as {filename}.";
                    UpdateStatus(msg);
                }
            }
        }
    }

    /// <summary>
    /// Guide button — opens the user guide HTML file in the default browser.
    /// </summary>
    private void OnGuide_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            // Try app base directory first (published exe), then current directory (debug)
            string guidePath = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory, "guide.html");

            if (!File.Exists(guidePath))
                guidePath = Path.Combine(Directory.GetCurrentDirectory(), "guide.html");

            if (File.Exists(guidePath))
            {
                Process.Start(new ProcessStartInfo(guidePath) { UseShellExecute = true });
                UpdateStatus("Opened user guide successfully.");
            }
            else
            {
                UpdateStatus("Error: Unable to open the user guide.");
            }
        }
        catch (Exception ex)
        {
            UpdateStatus($"Error: Unable to open the user guide. {ex.Message}");
        }
    }

    /// <summary>
    /// Batch Save button — generates multiple WAV files with unique random seeds.
    /// Each tone gets a fresh random base frequency for variety.
    /// </summary>
    private async void OnBatchSave_Click(object sender, RoutedEventArgs e)
    {
        if (!ValidateAndParseDuration(out float duration)) return;

        var mode = (FrequencyMode)FreqChoice.SelectedIndex;

        // Guard: ensure frequencies exist for the selected mode
        if (mode != FrequencyMode.DimensionalShift)
        {
            var freqResult = _frequencyManager.GetFrequencies(mode);
            if (freqResult.Frequencies.Length == 0 && !freqResult.IsBinaural)
            {
                UpdateStatus("Error: No frequencies available for the selected set.");
                return;
            }
        }

        // Ask user how many tones to generate (1-1000)
        var numDialog = new NumToneDialog();
        if (numDialog.ShowDialog() != true)
        {
            UpdateStatus("Batch save cancelled by user.");
            return;
        }
        int numTones = numDialog.NumTones;

        // Ask user where to save the files
        var folderDialog = new System.Windows.Forms.FolderBrowserDialog
        {
            Description = "Choose a directory to save the tones:",
        };
        if (folderDialog.ShowDialog() != System.Windows.Forms.DialogResult.OK)
        {
            UpdateStatus("Batch save cancelled by user.");
            return;
        }
        string saveDir = folderDialog.SelectedPath;

        _cts = new CancellationTokenSource();
        ToggleControls(isPlaying: true, showGauge: true);
        UpdateStatus($"Batch saving {numTones} tones to {saveDir}...");

        try
        {
            await Task.Run(async () =>
            {
                for (int i = 0; i < numTones; i++)
                {
                    if (_cts.Token.IsCancellationRequested) break;

                    // Each tone gets a fresh random base frequency
                    float baseFreq = DeriveBaseFreq(mode);
                    string filename = Path.Combine(saveDir, $"tone{i + 1}.wav");

                    await _soundPlayer.SaveToWavAsync(
                        filename, duration * 60f, baseFreq, 48000, _cts.Token,
                        updateStatus: UpdateStatus,
                        updateProgress: p =>
                        {
                            // Overall progress = (completed tones + current progress) / total
                            float overall = ((float)i + p) / numTones;
                            Dispatcher.InvokeAsync(() => Gauge.Value = overall * 100);
                        },
                        freqMode: mode);

                    await Dispatcher.InvokeAsync(() =>
                        UpdateStatus($"Saved tone {i + 1}/{numTones}: {Path.GetFileName(filename)}"));
                }
            }, _cts.Token);
        }
        catch (OperationCanceledException)
        {
            UpdateStatus("Batch save cancelled.");
        }
        catch (Exception ex)
        {
            UpdateStatus($"Batch save error: {ex.Message}");
        }
        finally
        {
            if (!_isClosing)
            {
                ToggleControls(isPlaying: false, showGauge: false);
                UpdateStatus("Batch save completed.");
            }
        }
    }

    /// <summary>
    /// Stop button — cancels the current operation and stops audio playback.
    /// </summary>
    private void OnStop_Click(object sender, RoutedEventArgs e)
    {
        _cts?.Cancel();
        _soundPlayer.StopPlayback();
        UpdateStatus("Stopping operation...");
    }

    /// <summary>
    /// Window closing — cancels any running operation and disposes audio resources.
    /// </summary>
    private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
    {
        _isClosing = true;
        _cts?.Cancel();
        _soundPlayer.StopPlayback();
        _soundPlayer.Dispose();
    }

    #endregion

    // Helper methods for device enumeration, mode name display, base frequency
    // derivation, UI control state toggling, status message output, and duration validation.
    #region UI Helpers

    /// <summary>
    /// Populate the output device dropdown with all available audio output devices.
    /// Uses NAudio's WaveOut.GetCapabilities to enumerate devices.
    /// Device -1 is the system default (Windows Sound Mapper).
    /// </summary>
    private void PopulateDeviceList()
    {
        DeviceChoice.Items.Clear();

        // Add the default device first (Windows Sound Mapper, device -1)
        DeviceChoice.Items.Add("Default Output Device");

        // Enumerate all available output devices
        for (int i = 0; i < WaveOut.DeviceCount; i++)
        {
            var caps = WaveOut.GetCapabilities(i);
            DeviceChoice.Items.Add(caps.ProductName);
        }

        DeviceChoice.SelectedIndex = 0;
    }

    /// <summary>
    /// Get the NAudio device number for the selected output device.
    /// Returns -1 for the default device, or 0-based index for specific devices.
    /// </summary>
    private int GetSelectedDeviceNumber()
    {
        // Index 0 = "Default Output Device" = NAudio device -1
        // Index 1+ = specific devices = NAudio device 0+
        int selected = DeviceChoice.SelectedIndex;
        return selected <= 0 ? -1 : selected - 1;
    }

    /// <summary>
    /// Convert a FrequencyMode enum to a human-readable display name.
    /// Most modes use the enum name directly; special modes get custom labels.
    /// </summary>
    private static string GetModeName(FrequencyMode mode) => mode switch
    {
        FrequencyMode.TripleHelixDna => "Triple Helix DNA Activation",
        FrequencyMode.TaygetanBinaural => "Taygetan Binaural",
        FrequencyMode.DimensionalShift => "Dimensional Shift",
        _ => mode.ToString(),
    };

    /// <summary>
    /// Derive the base frequency for a given mode.
    /// Modes 0-2: randomly pick from the mode's frequency list.
    /// Modes 3-5: use 432 Hz (sacred geometry ratios multiply the base).
    /// Mode 6 (Dimensional): always 432 Hz.
    /// </summary>
    private float DeriveBaseFreq(FrequencyMode mode)
    {
        if (mode == FrequencyMode.DimensionalShift) return 432f;

        var result = _frequencyManager.GetFrequencies(mode);

        // Binaural modes return frequency pairs — use the left ear frequency
        if (result.IsBinaural && result.BinauralPairs!.Length > 0)
        {
            var pair = result.BinauralPairs[Random.Shared.Next(result.BinauralPairs.Length)];
            return pair.Left;
        }

        // Standard modes — pick a random frequency from the set
        if (result.Frequencies.Length > 0)
            return result.Frequencies[Random.Shared.Next(result.Frequencies.Length)];

        // Fallback to 432 Hz (Lemurian keynote)
        return 432f;
    }

    /// <summary>
    /// Enable/disable UI controls based on whether an operation is running.
    /// Uses Enable/Disable pattern (not Show/Hide) for screen reader accessibility.
    /// Focus moves to Stop when playing, back to Play when idle.
    /// </summary>
    private void ToggleControls(bool isPlaying, bool showGauge = false)
    {
        bool isIdle = !isPlaying;

        // Enable/disable all input controls
        FreqChoice.IsEnabled = isIdle;
        DeviceChoice.IsEnabled = isIdle;
        DurationText.IsEnabled = isIdle;
        PlayBtn.IsEnabled = isIdle;
        SaveBtn.IsEnabled = isIdle;
        GuideBtn.IsEnabled = isIdle;
        BatchSaveBtn.IsEnabled = isIdle;
        StopBtn.IsEnabled = isPlaying;

        // Show/collapse stop button
        if (isPlaying)
            StopBtn.Visibility = Visibility.Visible;
        else
            StopBtn.Visibility = Visibility.Collapsed;

        // Show/collapse progress gauge
        if (showGauge)
        {
            Gauge.Value = 0;
            Gauge.Visibility = Visibility.Visible;
        }
        else
        {
            Gauge.Visibility = Visibility.Collapsed;
            Gauge.Value = 0;
        }

        // Focus management for screen reader — always move focus to the active button
        if (isPlaying)
            StopBtn.Focus();
        else
            PlayBtn.Focus();
    }

    /// <summary>
    /// Thread-safe status message output. Can be called from any thread.
    /// Dispatches to the UI thread if called from a background thread.
    /// </summary>
    private void UpdateStatus(string message)
    {
        if (_isClosing) return;

        if (Dispatcher.CheckAccess())
        {
            AppendAndAnnounce(message);
        }
        else
        {
            Dispatcher.InvokeAsync(() =>
            {
                if (!_isClosing)
                    AppendAndAnnounce(message);
            });
        }
    }

    /// <summary>
    /// Append a message to the native Win32 TextBox.
    /// Screen readers navigate this with arrow keys just like wx.TextCtrl in Python.
    /// </summary>
    private void AppendAndAnnounce(string message)
    {
        _statusTextBox.AppendText(message + Environment.NewLine);
    }

    /// <summary>
    /// Validate and parse the duration text field.
    /// Must be a positive number no greater than 60 minutes.
    /// Returns false and shows an error if invalid.
    /// </summary>
    private bool ValidateAndParseDuration(out float duration)
    {
        duration = 0;
        string text = DurationText.Text.Trim();

        if (string.IsNullOrEmpty(text))
        {
            UpdateStatus("Please enter a value.");
            DurationText.Focus();
            return false;
        }

        if (!float.TryParse(text, out duration) || duration <= 0)
        {
            UpdateStatus("Error: Duration must be positive.");
            DurationText.Focus();
            return false;
        }

        if (duration > 60)
        {
            UpdateStatus("Error: Duration must be <= 60 minutes.");
            DurationText.Focus();
            return false;
        }

        return true;
    }

    #endregion
}
