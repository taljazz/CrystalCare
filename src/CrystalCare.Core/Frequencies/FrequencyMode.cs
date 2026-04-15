namespace CrystalCare.Core.Frequencies;

/// <summary>
/// Frequency mode selection — replaces magic int indices throughout the pipeline.
/// Values match ComboBox order in MainWindow.xaml.
/// </summary>
public enum FrequencyMode
{
    Standard = 0,
    Solfeggio = 1,
    Fibonacci = 2,
    Pythagorean = 3,
    TripleHelixDna = 4,
    TaygetanBinaural = 5,
    DimensionalShift = 6,
}
