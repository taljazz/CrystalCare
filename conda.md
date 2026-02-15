# Working with Conda Environments - Guide for Claude Code

This guide documents how to effectively work with conda environments when developing Python projects with Claude Code.

## Why Use Conda Environments?

Conda environments provide isolated Python installations with specific package versions, preventing dependency conflicts between projects. Each project can have its own environment with exactly the packages it needs.

**Benefits:**
- **Isolation**: Each project has its own dependencies
- **Reproducibility**: Exact package versions can be specified
- **No conflicts**: Different projects can use different package versions
- **Easy cleanup**: Delete environment to remove all packages

---

## Basic Conda Commands

### Checking Conda Installation

```powershell
# Verify conda is installed
conda --version

# List all existing environments
conda env list
# or
conda info --envs
```

**Output example:**
```
# conda environments:
#
base                  *  C:\Users\YourName\miniconda3
yuansu                   C:\Users\YourName\.conda\envs\yuansu
```

The `*` indicates the currently active environment.

---

## Creating a New Environment

### Method 1: Basic Creation

```powershell
# Create environment with specific Python version
conda create -n myproject python=3.11 -y
```

**Parameters:**
- `-n myproject`: Name of the environment
- `python=3.11`: Python version to install
- `-y`: Auto-confirm (skip confirmation prompt)

### Method 2: Create with Initial Packages

```powershell
# Create environment with packages in one step
conda create -n myproject python=3.11 numpy pandas -y
```

### Method 3: From Environment File

```powershell
# Create from environment.yml file
conda env create -f environment.yml
```

**Example environment.yml:**
```yaml
name: myproject
channels:
  - defaults
dependencies:
  - python=3.11
  - numpy
  - pandas
  - pip
  - pip:
    - requests
    - some-pip-only-package
```

---

## Activating and Deactivating Environments

### Activation

```powershell
# Activate an environment
conda activate myproject
```

**After activation:**
- Your prompt changes to show `(myproject)` at the beginning
- `python` and `pip` commands now use the environment's installation
- All package installs go to this environment

### Deactivation

```powershell
# Deactivate current environment (return to base)
conda deactivate
```

---

## Installing Packages in Environments

### Installing with Conda

```powershell
# Install packages using conda
conda install numpy pandas matplotlib -y
```

### Installing with Pip (in Conda Environment)

**IMPORTANT:** Always activate the environment first!

```powershell
# Activate environment
conda activate myproject

# Install with pip (goes to environment)
pip install requests flask
```

### Installing Without Activation (Using conda run)

This is the **KEY TECHNIQUE** for Claude Code sessions:

```powershell
# Install packages without activating
conda run -n myproject pip install sounddevice numpy webrtcvad

# Multiple packages in one command
conda run -n myproject pip install package1 package2 package3
```

**Why use `conda run`?**
- Works even if environment isn't activated
- Ensures packages install to correct environment
- Prevents errors from wrong environment being active
- **Critical for Claude Code**: Claude can run commands without changing shell state

---

## Running Python Files from Conda Environments

### Method 1: Activate Then Run

```powershell
# Activate environment
conda activate myproject

# Run Python file
python myapp.py
```

### Method 2: Using conda run (Preferred for Claude Code)

```powershell
# Run Python file directly in environment (without activation)
conda run -n myproject python myapp.py
```

**Advantages of `conda run`:**
- No need to activate environment first
- Works from any directory
- Ensures correct environment is used
- **Perfect for Claude Code automated commands**

### Method 3: Full Path to Environment Python

```powershell
# Windows
C:\Users\YourName\.conda\envs\myproject\python.exe myapp.py

# Linux/Mac
~/.conda/envs/myproject/bin/python myapp.py
```

**When Method 3 is Useful:**
- When `conda` command is not available in your shell PATH
- In Git Bash on Windows where conda may not be initialized
- When you need absolute certainty about which Python is being used
- In Claude Code sessions where conda command may not be found

---

## Finding Conda Environment Locations

Conda environments can be stored in different locations depending on your installation and configuration. Here's how to find them:

### Common Environment Locations

**System-wide installation (Miniforge/Miniconda):**
```bash
# Windows
/c/ProgramData/miniforge3/envs/
C:\ProgramData\miniforge3\envs\

# Linux/Mac
/opt/miniconda3/envs/
```

**User-specific location (most common):**
```bash
# Windows (Git Bash format)
~/.conda/envs/
/c/Users/YourName/.conda/envs/

# Windows (PowerShell/CMD format)
C:\Users\YourName\.conda\envs\

# Linux/Mac
~/.conda/envs/
```

### How to Find Your Environments

**Method 1: List all environments with conda**
```bash
conda env list
# or
conda info --envs
```

This shows both environment names and their full paths:
```
# conda environments:
#
base                  *  C:\ProgramData\miniforge3
cc                       C:\Users\Thomas\.conda\envs\cc
yuansu                   C:\Users\Thomas\.conda\envs\yuansu
```

**Method 2: Check directory manually**
```bash
# Check system location first
ls -la /c/ProgramData/miniforge3/envs/

# Check user location
ls -la ~/.conda/envs/

# On Linux/Mac
ls -la ~/.conda/envs/
```

**Method 3: Find a specific environment's Python**
```bash
# Once you know the environment exists, verify Python location
ls -la ~/.conda/envs/cc/python.exe        # Windows
ls -la ~/.conda/envs/myproject/bin/python # Linux/Mac
```

### Running Python When Conda Command Not Available

If `conda run` doesn't work (common in Git Bash on Windows), use the direct path:

**Git Bash on Windows:**
```bash
# Format: ~/.conda/envs/ENVNAME/python.exe script.py
~/.conda/envs/cc/python.exe main.py

# Or with full Unix-style path
/c/Users/Thomas/.conda/envs/cc/python.exe main.py
```

**PowerShell/CMD on Windows:**
```powershell
# Format: C:\Users\YourName\.conda\envs\ENVNAME\python.exe script.py
C:\Users\Thomas\.conda\envs\cc\python.exe main.py
```

**Linux/Mac:**
```bash
# Format: ~/.conda/envs/ENVNAME/bin/python script.py
~/.conda/envs/myproject/bin/python app.py
```

### Running in Background (GUI Applications)

For wxPython or other GUI applications in Claude Code sessions:

```bash
# Run in background so GUI can open
~/.conda/envs/cc/python.exe main.py &

# Or let Claude Code handle background execution
~/.conda/envs/cc/python.exe main.py
```

### Troubleshooting: Environment Not Found

If you get "No such file or directory" errors:

```bash
# 1. Verify environment exists
conda env list

# 2. Check both possible locations
ls -la /c/ProgramData/miniforge3/envs/
ls -la ~/.conda/envs/

# 3. Look for the specific environment
ls -la ~/.conda/envs/cc/

# 4. Verify Python executable exists
ls -la ~/.conda/envs/cc/python.exe  # Windows
ls -la ~/.conda/envs/cc/bin/python  # Linux/Mac

# 5. If environment truly doesn't exist, create it
conda create -n cc python=3.11 -y
```

### Example: Complete Workflow

Here's how to find and run an environment when `conda run` isn't available:

```bash
# 1. Find where environments are stored
conda env list
# Output: cc    C:\Users\Thomas\.conda\envs\cc

# 2. Convert to Unix path for Git Bash (if needed)
# C:\Users\Thomas\.conda\envs\cc
# becomes: /c/Users/Thomas/.conda/envs/cc
# or use: ~/.conda/envs/cc

# 3. Verify Python exists
ls -la ~/.conda/envs/cc/python.exe

# 4. Run your application
~/.conda/envs/cc/python.exe main.py
```

---

## Managing Environments

### Listing Installed Packages

```powershell
# List packages in current environment
conda list

# List packages in specific environment (without activation)
conda list -n myproject

# List only pip-installed packages
conda run -n myproject pip list

# Search for specific package
conda list | Select-String -Pattern "numpy"
# or
conda run -n myproject pip list | Select-String -Pattern "numpy"
```

### Updating Packages

```powershell
# Update a specific package
conda update numpy -y

# Update all packages in environment
conda update --all -y
```

### Removing Packages

```powershell
# Remove a package
conda remove numpy -y

# Remove with pip
conda run -n myproject pip uninstall numpy -y
```

### Exporting Environment

```powershell
# Export to environment.yml (conda packages)
conda env export > environment.yml

# Export to requirements.txt (pip packages)
conda run -n myproject pip freeze > requirements.txt
```

### Cloning Environment

```powershell
# Create exact copy of environment
conda create --name myproject-backup --clone myproject
```

### Deleting Environment

```powershell
# Remove entire environment
conda env remove -n myproject -y
```

**WARNING:** This deletes all packages and the environment. Cannot be undone!

---

## Best Practices for Claude Code Sessions

### 1. **Always Specify Environment Name**

When working with Claude, always specify the environment explicitly:

```powershell
# Good - Explicit environment
conda run -n myproject python app.py

# Avoid - Relies on activation state
python app.py
```

### 2. **Use `conda run` for Installation**

Claude can install packages without activating:

```powershell
# Preferred for Claude Code
conda run -n myproject pip install newpackage

# Instead of:
conda activate myproject  # Claude can't persist activation
pip install newpackage     # Might install to wrong environment
```

### 3. **Verify Environment Before Running**

```powershell
# Check which environment would be used
conda run -n myproject python --version

# List installed packages to verify
conda run -n myproject pip list
```

### 4. **Document Environment in README**

Include in your project README:
- Environment name
- Python version
- How to create environment
- How to install dependencies
- How to run the application

**Example:**
```markdown
## Setup

```powershell
# Create environment
conda create -n myproject python=3.11 -y

# Install dependencies
conda run -n myproject pip install -r requirements.txt

# Run application
conda run -n myproject python app.py
```
```

### 5. **Keep requirements.txt Updated**

```powershell
# After adding new packages, update requirements.txt
conda run -n myproject pip freeze > requirements.txt
```

---

## Troubleshooting

### Environment Not Found

```powershell
# Error: Could not find conda environment: myproject
# Solution: List environments to verify name
conda env list
```

### Wrong Python/Packages Being Used

```powershell
# Check which Python is being used
conda run -n myproject python -c "import sys; print(sys.executable)"

# Expected output:
# C:\Users\YourName\.conda\envs\myproject\python.exe
```

### Package Install Fails

```powershell
# Try installing with conda instead of pip
conda install -n myproject package_name -y

# If package not in conda, use pip
conda run -n myproject pip install package_name

# Update pip first if needed
conda run -n myproject pip install --upgrade pip
```

### Environment Activation Fails

```powershell
# Initialize conda for PowerShell (one-time setup)
conda init powershell

# Restart PowerShell after running conda init
```

### "Solving environment" Takes Forever

```powershell
# Use mamba (faster conda alternative)
conda install mamba -n base -c conda-forge -y
mamba install package_name -y

# Or use pip for that package
conda run -n myproject pip install package_name
```

---

## Quick Reference - Common Workflows

### Creating New Project Environment

```powershell
# 1. Create environment
conda create -n myproject python=3.11 -y

# 2. Install common packages
conda run -n myproject pip install numpy pandas matplotlib requests

# 3. Save requirements
conda run -n myproject pip freeze > requirements.txt

# 4. Run your code
conda run -n myproject python app.py
```

### Reproducing Environment on Another Machine

```powershell
# On original machine - Export
conda run -n myproject pip freeze > requirements.txt

# On new machine - Import
conda create -n myproject python=3.11 -y
conda run -n myproject pip install -r requirements.txt
```

### Checking Environment Health

```powershell
# Verify environment exists
conda env list

# Check Python version
conda run -n myproject python --version

# List all packages
conda run -n myproject pip list

# Check for specific package
conda run -n myproject pip show numpy
```

---

## Special Note: conda run vs Activation

### When to Use `conda run`

✅ **Use for Claude Code sessions:**
- Claude executing commands
- Automated scripts
- CI/CD pipelines
- When you want explicit environment control

```powershell
conda run -n myproject python app.py
```

### When to Use Activation

✅ **Use for interactive development:**
- Manual testing
- Interactive Python sessions
- Multiple commands in same environment

```powershell
conda activate myproject
python
>>> import numpy
>>> # Interactive work
```

---

## Example: Complete Project Setup

Here's a complete example of setting up the Yuansu project:

```powershell
# 1. Create environment with Python 3.11
conda create -n yuansu python=3.11 -y

# 2. Install all dependencies
conda run -n yuansu pip install sounddevice numpy webrtcvad faster-whisper requests piper-tts wxPython matplotlib psutil pynvml GPUtil wmi pywin32

# 3. Verify installation
conda run -n yuansu pip list | Select-String -Pattern "wxPython|piper|whisper"

# 4. Run the application
conda run -n yuansu python yuansu.py

# 5. Alternative: Activate and run (for interactive development)
conda activate yuansu
python yuansu.py
```

---

## Integration with Claude Code

### Why `conda run` is Perfect for Claude

Claude Code can execute commands but cannot maintain shell state (like active environments) between commands. Using `conda run` solves this:

**Problem:**
```powershell
# Command 1 (Claude runs this)
conda activate myproject

# Command 2 (Claude runs this in NEW shell - activation lost!)
python app.py  # ERROR: Wrong Python!
```

**Solution:**
```powershell
# Single command that works every time
conda run -n myproject python app.py
```

### Recommended Pattern for Claude Sessions

1. **Installation:**
   ```powershell
   conda run -n myproject pip install package1 package2
   ```

2. **Running code:**
   ```powershell
   conda run -n myproject python app.py
   ```

3. **Verification:**
   ```powershell
   conda run -n myproject python -c "import package1; print(package1.__version__)"
   ```

---

## Summary - Key Takeaways

1. **Create environments** for each project: `conda create -n project python=3.11 -y`
2. **Use `conda run`** for commands in Claude Code sessions: `conda run -n project pip install ...`
3. **List environments** to verify: `conda env list`
4. **Install packages** explicitly: `conda run -n project pip install package`
5. **Run Python files** explicitly: `conda run -n project python app.py`
6. **Export dependencies**: `conda run -n project pip freeze > requirements.txt`
7. **Delete environments** when done: `conda env remove -n project -y`

**Golden Rule for Claude Code:**
> Always specify the environment explicitly using `conda run -n envname` to ensure commands execute in the correct environment.

---

## Additional Resources

- [Conda Documentation](https://docs.conda.io/)
- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
- [Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [Using Pip in Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-non-conda-packages)
