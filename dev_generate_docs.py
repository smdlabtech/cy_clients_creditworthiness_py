import os
import re
import ast
from pathlib import Path

def extract_info_from_file(file_path):
    """
    Extrait les informations pertinentes (docstrings, fonctions, classes) d'un fichier Python,
    et le contenu pour les fichiers CSS et JavaScript, tout en ignorant les types de fichiers spécifiés.
    
    Args:
        file_path (str): Le chemin complet vers le fichier.
        
    Returns:
        dict: Un dictionnaire contenant les informations extraites.
    """
    ignored_extensions = ['.gitattributes', '.gitignore', 'LICENSE', '.md', '.txt', '.db', '.pdf']
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() in ignored_extensions:
        return {}  # Retourne un dictionnaire vide pour les fichiers ignorés
    
    info_dict = {'type': file_extension}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            if file_path.endswith('.py'):
                tree = ast.parse(file.read(), filename=file_path)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        name = node.name
                        docstring = ast.get_docstring(node)
                        if docstring:
                            info_dict[name] = docstring.strip()
                        else:
                            info_dict[name] = "No docstring found."
            elif file_path.endswith(('.css', '.js')):
                content = file.read()
                info_dict['content'] = content
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {file_path}: {e}")
    return info_dict

def generate_documentation_md(directory_path, output_file):
    """
    Génère un fichier Markdown contenant la documentation de tous les fichiers dans un répertoire,
    en ignorant les répertoires spécifiés et les types de fichiers spécifiques.
    
    Args:
        directory_path (str): Le chemin du répertoire contenant les fichiers.
        output_file (str): Le chemin du fichier de sortie Markdown.
    """
    excluded_dirs = [".", "venv", "env", ".git", "__pycache__","_data", "data", "data_lcl_pdf","assets", "archives"]
    with open(output_file, 'w', encoding='utf-8') as md_file:
        for root, dirs, files in os.walk(directory_path):
            # Ignore directories that start with a dot or match one of the excluded patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in excluded_dirs]
            
            for file in files:
                file_path = os.path.join(root, file)
                info_dict = extract_info_from_file(file_path)
                if info_dict:
                    filename = os.path.basename(file)
                    md_file.write(f"# File {filename} ({info_dict['type']})\n")
                    for key, value in info_dict.items():
                        if key!= 'type':
                            md_file.write(f" - **{key}** : {value}\n")
                    md_file.write("\n")

if __name__ == "__main__":
    # Obtenez le répertoire courant du script
    current_dir = Path(__file__).resolve().parent
    
    # Utilisez current_dir comme directory_path pour générer la documentation
    directory_path = str(current_dir)
    output_file = "dev_documentations.md"
    generate_documentation_md(directory_path, output_file)
