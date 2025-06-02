import os
import sys
import subprocess
import venv

def create_venv(venv_dir="venv"):
    if not os.path.isdir(venv_dir):
        print(f"🛠️ Création de l'environnement virtuel dans ./{venv_dir} ...")
        venv.create(venv_dir, with_pip=True)
    else:
        print(f"✅ Environnement virtuel {venv_dir} déjà présent.")

def install_requirements(venv_dir="venv", requirements_file="requirements.txt"):
    print("📦 Installation des dépendances depuis requirements.txt ...")
    if os.name == "nt":
        pip_path = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        pip_path = os.path.join(venv_dir, "bin", "pip")
    
    cmd = [pip_path, "install", "-r", requirements_file]
    subprocess.check_call(cmd)
    print("✅ Installation terminée.")

def main():
    create_venv()
    install_requirements()
    print("\n🎉 Tout est prêt ! Tu peux maintenant lancer l'application avec :")
    print("   python run.py\n")

if __name__ == "__main__":
    main()
