using System.Windows;

namespace CrystalCare;

/// <summary>
/// Interaction logic for App.xaml.
///
/// MainWindow is created and shown manually in OnStartup rather than via
/// XAML-based StartupUri because the StartupUri lookup path fails in a
/// single-file self-contained EXE. The compressed manifest gets queried with
/// a lowercased resource name ("mainwindow.xaml") while the actual embedded
/// resource is proper-cased ("MainWindow.xaml"), causing System.IO.IOException
/// at launch. Direct instantiation in OnStartup avoids the resource lookup.
/// </summary>
public partial class App : System.Windows.Application
{
    /// <summary>
    /// Application startup hook — replaces the XAML StartupUri mechanism.
    /// Instantiates and shows the main window directly so the WPF resource
    /// lookup never runs (avoiding the single-file case-sensitivity bug).
    /// </summary>
    protected override void OnStartup(StartupEventArgs e)
    {
        // Catch any startup exception and surface it via a message box so
        // a crash during MainWindow construction is visible to the user
        // rather than appearing as a silently-running process with no UI.
        try
        {
            // Let WPF do its normal startup bookkeeping first
            base.OnStartup(e);

            // Create the main window manually — same effect as
            // StartupUri="MainWindow.xaml" would have, minus the broken
            // resource lookup that crashes single-file builds.
            var mainWindow = new MainWindow();

            // Explicitly set Application.MainWindow so WPF's shutdown mode
            // (OnLastWindowClose by default) properly tracks the window.
            this.MainWindow = mainWindow;

            mainWindow.Show();
        }
        catch (System.Exception ex)
        {
            // Use the fully qualified WPF MessageBox name. The project also
            // brings in System.Windows.Forms (for the WindowsFormsHost in
            // MainWindow), and an unqualified "MessageBox" reference would
            // be ambiguous between WPF and WinForms — qualifying it picks
            // the WPF version explicitly.
            System.Windows.MessageBox.Show(
                $"CrystalCare failed to start:{System.Environment.NewLine}{System.Environment.NewLine}{ex.Message}",
                "CrystalCare startup error",
                MessageBoxButton.OK,
                MessageBoxImage.Error);

            // Re-throw so the runtime / Windows event log also see the
            // exception (helpful when diagnosing field issues).
            throw;
        }
    }
}
