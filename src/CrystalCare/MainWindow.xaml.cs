using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Automation;
using System.Windows.Automation.Peers;
using CrystalCare.Audio;
using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Generation;
using Microsoft.Win32;

namespace CrystalCare;

/// <summary>
/// Debug logger that writes to a file alongside the exe.
/// </summary>
internal static class DebugLog
{
    private static readonly string LogPath = Path.Combine(
        AppDomain.CurrentDomain.BaseDirectory, "crystalcare_debug.log");
    private static readonly object Lock = new();

    public static void Log(string message)
    {
        try
        {
            lock (Lock)
            {
                File.AppendAllText(LogPath,
                    $"[{DateTime.Now:HH:mm:ss.fff}] [Thread {Environment.CurrentManagedThreadId}] {message}{Environment.NewLine}");
            }
        }
        catch { }
    }
}

/// <summary>
/// Main application window — WPF GUI with full screen reader accessibility.
/// Port of CrystalCareFrame from main.py.
/// </summary>
public partial class MainWindow : Window
{
    private static readonly Dictionary<int, string> FrequencyModes = new()
    {
        [0] = "Standard",
        [1] = "Solfeggio",
        [2] = "Fibonacci",
        [3] = "Pythagorean",
        [4] = "Triple Helix DNA Activation",
        [5] = "Taygetan Binaural",
        [6] = "Dimensional Shift",
    };

    /// <summary>
    /// Derive base frequency from the frequency manager, matching Python logic.
    /// For modes 0-2: randomly pick from the returned frequency list.
    /// For modes 3-5: use 432 Hz (sacred geometry uses ratios * base).
    /// For mode 6 (dimensional): use 432 Hz.
    /// </summary>
    private float DeriveBaseFreq(int selection)
    {
        if (selection == 6) return 432f;

        var result = _frequencyManager.GetFrequencies(selection);
        if (result.IsBinaural && result.BinauralPairs!.Length > 0)
        {
            var pair = result.BinauralPairs[Random.Shared.Next(result.BinauralPairs.Length)];
            return pair.Left;
        }

        if (result.Frequencies.Length > 0)
            return result.Frequencies[Random.Shared.Next(result.Frequencies.Length)];

        return 432f;
    }

    private readonly FrequencyManager _frequencyManager = new();
    private readonly SoundGenerator _soundGenerator;
    private readonly SoundPlayer _soundPlayer;
    private CancellationTokenSource? _cts;
    private bool _isClosing;

    // Native Win32 TextBox for reliable screen reader navigation
    private readonly System.Windows.Forms.TextBox _statusTextBox;

    public MainWindow()
    {
        InitializeComponent();

        // Create a native Win32 multiline read-only TextBox — same as wx.TextCtrl
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

        _soundGenerator = new SoundGenerator(_frequencyManager);
        _soundPlayer = new SoundPlayer(_soundGenerator);

        UpdateStatus("CrystalCare ready. Select a frequency set and duration, then press Play.");
    }

    // ========== EVENT HANDLERS ==========

    private async void OnPlay_Click(object sender, RoutedEventArgs e)
    {
        DebugLog.Log("=== OnPlay_Click START ===");
        if (!ValidateAndParseDuration(out float duration)) return;
        int selection = FreqChoice.SelectedIndex;
        bool dimensionalMode = selection == 6;
        float baseFreq = DeriveBaseFreq(selection);
        DebugLog.Log($"  selection={selection} baseFreq={baseFreq} duration={duration}min dimensional={dimensionalMode}");

        _cts = new CancellationTokenSource();
        ToggleControls(isPlaying: true);
        UpdateStatus("Resonating...");

        try
        {
            await _soundPlayer.PlayStreamAsync(
                duration * 60f, baseFreq, 48000, _cts.Token,
                updateStatus: UpdateStatus,
                dimensionalMode: dimensionalMode,
                freqSelection: selection);
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            UpdateStatus($"Error: {ex.Message}");
            DebugLog.Log($"PLAY EXCEPTION: {ex}");
        }
        finally
        {
            if (!_isClosing)
            {
                ToggleControls(isPlaying: false);
                string modeName = FrequencyModes.GetValueOrDefault(selection, "");
                string msg = selection is 4 or 5 or 6
                    ? $"{modeName} completed."
                    : "Playback completed.";
                UpdateStatus(msg);
            }
        }
    }

    private async void OnSave_Click(object sender, RoutedEventArgs e)
    {
        if (!ValidateAndParseDuration(out float duration)) return;

        int selection = FreqChoice.SelectedIndex;

        // Check frequencies available (matching Python guard)
        if (selection != 6)
        {
            var freqResult = _frequencyManager.GetFrequencies(selection);
            if (freqResult.Frequencies.Length == 0 && !freqResult.IsBinaural)
            {
                UpdateStatus("Error: No frequencies available for the selected set.");
                return;
            }
        }

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
        bool dimensionalMode = selection == 6;
        float baseFreq = DeriveBaseFreq(selection);

        _cts = new CancellationTokenSource();
        ToggleControls(isPlaying: true, showGauge: true);
        UpdateStatus("Resonating and saving to file...");

        try
        {
            await _soundPlayer.SaveToWavAsync(
                filename, duration * 60f, baseFreq, 48000, _cts.Token,
                updateStatus: UpdateStatus,
                updateProgress: p => Dispatcher.InvokeAsync(() => Gauge.Value = p * 100),
                dimensionalMode: dimensionalMode,
                freqSelection: selection);
        }
        catch (OperationCanceledException)
        {
            UpdateStatus("Save cancelled.");
        }
        catch (Exception ex)
        {
            UpdateStatus($"Save error: {ex.Message}");
            DebugLog.Log($"SAVE EXCEPTION: {ex}");
        }
        finally
        {
            if (!_isClosing)
            {
                ToggleControls(isPlaying: false, showGauge: false);
                if (!(_cts?.IsCancellationRequested ?? true))
                {
                    string modeName = FrequencyModes.GetValueOrDefault(selection, "");
                    string msg = selection is 4 or 5 or 6
                        ? $"{modeName} saved as {filename}."
                        : $"Resonance saved as {filename}.";
                    UpdateStatus(msg);
                }
            }
        }
    }

    private void OnGuide_Click(object sender, RoutedEventArgs e)
    {
        try
        {
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

    private async void OnBatchSave_Click(object sender, RoutedEventArgs e)
    {
        if (!ValidateAndParseDuration(out float duration)) return;

        int selection = FreqChoice.SelectedIndex;

        // Check frequencies available
        if (selection != 6)
        {
            var freqResult = _frequencyManager.GetFrequencies(selection);
            if (freqResult.Frequencies.Length == 0 && !freqResult.IsBinaural)
            {
                UpdateStatus("Error: No frequencies available for the selected set.");
                return;
            }
        }

        // Number of tones dialog
        var numDialog = new NumToneDialog();
        if (numDialog.ShowDialog() != true)
        {
            UpdateStatus("Batch save cancelled by user.");
            return;
        }
        int numTones = numDialog.NumTones;

        // Directory selection
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

        bool dimensionalMode = selection == 6;

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

                    float baseFreq = DeriveBaseFreq(selection);
                    string filename = Path.Combine(saveDir, $"tone{i + 1}.wav");

                    await _soundPlayer.SaveToWavAsync(
                        filename, duration * 60f, baseFreq, 48000, _cts.Token,
                        updateStatus: UpdateStatus,
                        updateProgress: p =>
                        {
                            float overall = ((float)i + p) / numTones;
                            Dispatcher.InvokeAsync(() => Gauge.Value = overall * 100);
                        },
                        dimensionalMode: dimensionalMode,
                        freqSelection: selection);

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
            DebugLog.Log($"BATCH SAVE EXCEPTION: {ex}");
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

    private void OnStop_Click(object sender, RoutedEventArgs e)
    {
        DebugLog.Log("=== OnStop_Click ===");
        _cts?.Cancel();
        _soundPlayer.StopPlayback();
        UpdateStatus("Stopping operation...");
    }

    private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
    {
        _isClosing = true;
        _cts?.Cancel();
        _soundPlayer.StopPlayback();
        _soundGenerator.Dispose();
        _soundPlayer.Dispose();
    }

    // ========== UI HELPERS ==========

    private void ToggleControls(bool isPlaying, bool showGauge = false)
    {
        DebugLog.Log($"ToggleControls: isPlaying={isPlaying} showGauge={showGauge}");
        bool isIdle = !isPlaying;

        FreqChoice.IsEnabled = isIdle;
        DurationText.IsEnabled = isIdle;
        PlayBtn.IsEnabled = isIdle;
        SaveBtn.IsEnabled = isIdle;
        GuideBtn.IsEnabled = isIdle;
        BatchSaveBtn.IsEnabled = isIdle;
        StopBtn.IsEnabled = isPlaying;

        if (isPlaying)
            StopBtn.Visibility = Visibility.Visible;
        else
            StopBtn.Visibility = Visibility.Collapsed;

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

        // Focus management for screen reader
        if (isPlaying)
            StopBtn.Focus();
        else
            PlayBtn.Focus();
    }

    private void UpdateStatus(string message)
    {
        DebugLog.Log($"UpdateStatus: \"{message}\" | onUIThread={Dispatcher.CheckAccess()}");

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
    /// Append text to the native Win32 TextBox.
    /// Screen readers navigate this with arrow keys just like wx.TextCtrl in Python.
    /// </summary>
    private void AppendAndAnnounce(string message)
    {
        _statusTextBox.AppendText(message + Environment.NewLine);

        DebugLog.Log($"  Appended: \"{message}\" (len={_statusTextBox.TextLength})");
    }

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
}

/// <summary>
/// Simple dialog to get number of tones for batch save.
/// Matches Python's wx.NumberEntryDialog.
/// </summary>
public class NumToneDialog : Window
{
    private readonly System.Windows.Controls.TextBox _input;
    public int NumTones { get; private set; } = 1;

    public NumToneDialog()
    {
        Title = "Batch Save";
        Width = 300;
        Height = 150;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        ResizeMode = ResizeMode.NoResize;

        var panel = new System.Windows.Controls.StackPanel { Margin = new Thickness(10) };

        panel.Children.Add(new System.Windows.Controls.TextBlock
        {
            Text = "Enter the number of tones to save:",
            Margin = new Thickness(0, 0, 0, 8),
        });

        _input = new System.Windows.Controls.TextBox
        {
            Text = "1",
            Margin = new Thickness(0, 0, 0, 8),
        };
        _input.SelectAll();
        panel.Children.Add(_input);

        var btnPanel = new System.Windows.Controls.StackPanel
        {
            Orientation = System.Windows.Controls.Orientation.Horizontal,
            HorizontalAlignment = System.Windows.HorizontalAlignment.Right,
        };
        var okBtn = new System.Windows.Controls.Button
        {
            Content = "OK", Width = 75, Margin = new Thickness(5, 0, 0, 0), IsDefault = true,
        };
        okBtn.Click += (_, _) =>
        {
            if (int.TryParse(_input.Text, out int n) && n >= 1 && n <= 1000)
            {
                NumTones = n;
                DialogResult = true;
            }
        };
        var cancelBtn = new System.Windows.Controls.Button
        {
            Content = "Cancel", Width = 75, Margin = new Thickness(5, 0, 0, 0), IsCancel = true,
        };
        btnPanel.Children.Add(okBtn);
        btnPanel.Children.Add(cancelBtn);
        panel.Children.Add(btnPanel);

        Content = panel;
    }
}
