# styles_app.py
import os


def input_css(css_file):
    """
    Loads and applies a CSS file to a Streamlit application.
    
    Args:
    css_file (str): Name of CSS file to load.
    """
    
    # # Construire le chemin complet vers le fichier CSS en utilisant le nom du fichier passé en argument
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".", "assets", "css", css_file)

    with open(css_path) as f:
        css = f.read()

    # Retourne le CSS sous forme de chaîne pour être utilisé par un script Streamlit
    return css


