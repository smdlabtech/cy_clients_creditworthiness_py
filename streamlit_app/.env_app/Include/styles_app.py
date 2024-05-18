# styles_app.py
import os
import streamlit as st


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

# Chargemement d'image dans l'application
def load_image(image_file, caption=None, use_column_width=True):
    """
    Loads and displays an image in a Streamlit application.
    
    Args:
    image_file (str): Name of the image file to load.
    caption (str): Optional caption to display below the image.
    use_column_width (bool): Whether to use the column width for the image.
    """
    # Construire le chemin complet vers le fichier image en utilisant le nom du fichier passé en argument
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".", "assets", "img", image_file)
    return image_path