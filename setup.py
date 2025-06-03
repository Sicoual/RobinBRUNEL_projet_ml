import os
import sys
import subprocess
import venv
import platform

# Versions compatibles avec TF 2.12.0
COMPATIBLE_PYTHON_VERSIONS = [(3,8), (3,9), (3,10), (3,11)]

def check_python_version():
    major, minor = sys.version_info[:2]
    if (major, minor) not in COMPATIBLE_PYTHON_VERSIONS:
        print(f"❌ Python {major}.{minor} n'est pas compatible avec TensorFlow 2.12.0.")
        print(f"Veuillez utiliser Python {', '.join([f'{v[0]}.{v[1]}' for v in COMPATIBLE_PYTHON_VERSIONS])}.")
        sys.exit(1)

def create_venv(venv_dir="venv"):
    if not os.path.isdir(venv_dir):
        print(f"🛠️ Création de l'environnement virtuel dans ./{venv_dir} ...")
        venv.EnvBuilder(with_pip=True).create(venv_dir)
    else:
        print(f"✅ Environnement virtuel {venv_dir} déjà présent.")

def install_requirements(venv_dir="venv", requirements_file="requirements.txt"):
    print("📦 Installation des dépendances depuis requirements.txt ...")

    if not os.path.isfile(requirements_file):
        print(f"❌ Le fichier {requirements_file} est introuvable.")
        sys.exit(1)

    if os.name == "nt":
        python_path = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python_path = os.path.join(venv_dir, "bin", "python")

    cmd = [python_path, "-m", "pip", "install", "-r", requirements_file]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation des dépendances : {e}")
        sys.exit(1)
    print("✅ Installation terminée.")

def main():
    check_python_version()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_file = os.path.join(script_dir, "requirements.txt")

    create_venv()
    install_requirements(requirements_file=requirements_file)

    print("\n🎉 Tout est prêt ! Tu peux maintenant lancer l'application avec :")
    print("   python run.py\n")

if __name__ == "__main__":
    main()
