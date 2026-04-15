using System.Windows;

namespace CrystalCare;

/// <summary>
/// Simple dialog to get number of tones for batch save.
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
