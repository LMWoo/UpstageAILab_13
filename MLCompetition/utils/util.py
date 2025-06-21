
import importlib
import os
import eli5
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ace_tools_open as tools
from bs4 import BeautifulSoup

def dynamic_preprocessing_import(root_module: str, sub_path: str):
    *submodules, class_name = sub_path.split(".")
    full_module_path = ".".join([root_module] + submodules)
    module = importlib.import_module(full_module_path)
    return getattr(module, class_name)

def save_eli5_html_to_image(save_path, perm, feature_names):
    html = eli5.format_as_html(eli5.explain_weights(perm, feature_names=feature_names))
    soup = BeautifulSoup(html, 'html.parser')

    data = []
    table = soup.find('table', {'class': 'eli5-weights'})

    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if len(cols) != 2:
            continue
        weight_text = cols[0].get_text(strip=True).replace('\n', '')
        feature = cols[1].get_text(strip=True)
                    
        if '±' in weight_text:
            mean, std = weight_text.split('±')
            mean = float(mean.strip().replace(',', ''))
            std = float(std.strip().replace(',', ''))
        else:
            mean = float(weight_text.strip().replace(',',''))
            std = None
                
        data.append((feature, mean, std))

    df = pd.DataFrame(data, columns=['Feature', 'Importance', 'StdDev'])

    df = df.sort_values('Importance', ascending=False).reset_index(drop=True)

    tools.display_dataframe_to_user(name="Permutation Importance Table", dataframe=df)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, y='Feature', x='Importance')
    plt.title('Permutation Importance')
    plt.xlabel('Decrease in Model Score')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()