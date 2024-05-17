# mod_styles_app.py
import os

def input_css(css_file):
    """
    Charge et applique un fichier CSS à une application Streamlit.
    
    Args:
    css_file (str): Nom du fichier CSS à charger.
    """
    # # Chemin du répertoire courant
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # # Construire le chemin complet vers le fichier CSS en utilisant le nom du fichier passé en argument
    # css_path = os.path.join(current_dir, "assets/css/", css_file)
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".", "assets", "css", css_file)

    with open(css_path) as f:
        css = f.read()

    # Retourne le CSS sous forme de chaîne pour être utilisé par un script Streamlit
    return css


