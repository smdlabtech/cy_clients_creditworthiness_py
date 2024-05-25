import os
import streamlit as st
from PIL import Image
import io


def load_img(image_file, caption=None, use_column_width=True, width=None, output_format=None):
    """
    Charge et affiche une image dans une application Streamlit.
    
    Args:
    image_file (str): Nom du fichier image à charger.
    caption (str): Légende facultative à afficher sous l'image.
    use_column_width (bool): Utiliser ou non la largeur de colonne pour l'image.
    width (int): Largeur de l'image en pixels.
    height (int): Hauteur de l'image en pixels.
    output_format (str): Format de sortie de l'image (par exemple, 'PNG', 'JPEG').
    """
    # Construire le chemin complet vers le fichier image en utilisant le nom du fichier passé en argument
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "img", image_file)
    return st.image(image_path, caption=caption, use_column_width=use_column_width, width=width, output_format=output_format)


def load_css(css_file):
    """
    Charge et applique un fichier CSS à une application Streamlit.
    
    Args:
    css_file (str): Nom du fichier CSS à charger.
    """
    # Construire le chemin complet vers le fichier CSS en utilisant le nom du fichier passé en argument
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "css", css_file)
    if not os.path.exists(css_path):
        raise FileNotFoundError(f"File '{css_file}' not found at '{css_path}'")
    
    with open(css_path) as f:
        css = f.read()
    
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def load_html(file_name):
    """
    Charge le contenu HTML à partir du nom de fichier donné.
    
    Args:
    file_name (str): Nom du fichier HTML à charger.

    Returns:
    str: Le contenu HTML en tant que chaîne de caractères.
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "html", file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_name}' not found at '{file_path}'")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    return content


def load_js(file_name):
    """
    Charge le contenu JavaScript à partir du nom de fichier donné.
    
    Args:
    file_name (str): Nom du fichier JavaScript à charger.

    Returns:
    str: Le contenu JavaScript en tant que chaîne de caractères.
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "js", file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_name}' not found at '{file_path}'")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    return content
