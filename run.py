import os
import sys
import subprocess
import platform

def get_venv_python():
    """Retourne le chemin python dans le venv selon l'OS"""
    base = os.path.join(os.getcwd(), "venv")
    if platform.system() == "Windows":
        return os.path.join(base, "Scripts", "python.exe")
    else:
        return os.path.join(base, "bin", "python")

def run_streamlit():
    python_venv = get_venv_python()
    if not os.path.exists(python_venv):
        print("⚠️ L'environnement virtuel n'existe pas. Merci de lancer 'python setup.py' avant.")
        sys.exit(1)

    # Commande à lancer : python -m streamlit run pages/app.py
    cmd = [python_venv, "-m", "streamlit", "run", "pages/app.py"]

    print("🚀 Lancement de l'application avec :")
    print(" ".join(cmd))
    # On lance la commande, on transmet stdout/stderr pour voir le retour
    subprocess.run(cmd)

if __name__ == "__main__":
    run_streamlit()
