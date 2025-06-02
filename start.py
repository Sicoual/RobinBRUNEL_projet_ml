import os
import sys
import subprocess
import platform

def get_venv_python():
    if platform.system() == "Windows":
        return os.path.join("venv", "Scripts", "python.exe")
    else:
        return os.path.join("venv", "bin", "python")

def main():
    python_executable = get_venv_python()
    if not os.path.exists(python_executable):
        print("❌ Environnement virtuel non trouvé. Crée-le avec setup.py ou setup script.")
        sys.exit(1)

    # Commande pour lancer Streamlit avec python du venv
    cmd = [python_executable, "-m", "streamlit", "run", "app-ml-vin/app.py"]

    print(f"🚀 Lancement de Streamlit avec : {cmd}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
