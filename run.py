#since bash scripts don't seem to work, here's a python script to automate dependencies
import os
import sys
import subprocess

def in_venv():
    # Reliable detection: sys.prefix != sys.base_prefix (and older Python via real_prefix)
    return (
        getattr(sys, 'real_prefix', None) is not None or
        sys.prefix != sys.base_prefix
    )

def ensure_venv(path='ENV'):
    if in_venv():
        print("Already inside a virtual environment.")
        return

    if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, 'pyvenv.cfg')):
        print(f"Creating virtual environment at ./{path}")
        subprocess.run([sys.executable, '-m', 'venv', path], check=True)
        print(f"Loading opencv NOTE: it may give you a warning, so far it has not caused any problems though.")
        subprocess.run("module load gcc cuda opencv/4.11.0 && . ENV/bin/activate && pip install --upgrade pip && pip install --no-index -r requirements-sharc.txt", shell=True)
        requirements_path = "requirements-sharc.txt"
        try:
          venv_pip = os.path.join(path, 'bin', 'pip')
          print("Installing dependencies…")
          #subprocess.run(". ENV/bin/activate", shell=True)
          #subprocess.run([venv_pip, 'install', '--no-index', '-r', 'requirements-sharc.txt'], check=True)
          #print("Installed dependencies via venv pip.")
        except subprocess.CalledProcessError:
          print("Failed to install one or more packages.")
          sys.exit(1)
            
    else:
        print(f"Found existing virtual environment in ./{path}.")
    print(f"Start up venv using . ENV/bin/activate. NOTE: if you have an issue with import cv2 then, while outside of venv (enter >deactivate to exit) use >module load opencv cuda")



if __name__ == '__main__':
    ensure_venv()
# # Path to your requirements.txt file
# requirements_path = "requirements-sharc.txt"

# # Check if the file exists
# if not os.path.exists(requirements_path):
#     print(f"Error: {requirements_path} not found.")
#     sys.exit(1)
    
# subprocess.run(".", shell=True)
#NOTE: 
# Install using pip
# try:
#     print(f"Installing dependencies from {requirements_path}...")
#     subprocess.run("module load gcc cuda opencv/4.11.0", shell=True)
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "-r",requirements_path, "--no-index"])
#     subprocess.run("pip install --no-index torch torchvision torchtext torchaudio", shell=True) 
#     print("All packages installed successfully.")
# except subprocess.CalledProcessError:
#     print("Failed to install one or more packages.")
#     sys.exit(1)
