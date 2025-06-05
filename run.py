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

def run_streamlit(script_path="pages/app.py"):
    python_venv = get_venv_python()
    if not os.path.exists(python_venv):
        print("⚠️ L'environnement virtuel n'existe pas. Merci de lancer 'python setup.py' avant.")
        sys.exit(1)

    cmd = [python_venv, "-m", "streamlit", "run", script_path]
    print("🚀 Lancement de l'application avec :")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'exécution de Streamlit : {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Streamlit n'est pas installé dans l'environnement virtuel. Merci de vérifier les dépendances.")
        sys.exit(1)

if __name__ == "__main__":
    # Permet de passer un chemin en argument si besoin
    if len(sys.argv) > 1:
        run_streamlit(sys.argv[1])
    else:
        run_streamlit()
