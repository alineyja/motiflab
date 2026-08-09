import py3Dmol
from pathlib import Path

def generate_ctcf_3d_view(save_path: str | Path):
    pdb_id = "5YEL"
    
    # Создаем холст 3D-сцены (разрешение 800x600)
    view = py3Dmol.view(query=f"pdb:{pdb_id}", width=800, height=600)
    
    # 1. Настраиваем фон
    view.setBackgroundColor('white')
    
    # 2. Рендерим белок (CTCF) в виде красивых "лент" (cartoon)
    # Раскрашиваем белок в цвет 'chain A' (синий/зеленый)
    view.setStyle({'chain': 'A'}, {'cartoon': {'color': 'spectrum'}})
    
    # 3. Рендерим ДНК (Цепи B и C) в виде атомарных сфер и палочек
    view.setStyle({'chain': 'B'}, {'stick': {'colorscheme': 'nucleic'}})
    view.setStyle({'chain': 'C'}, {'stick': {'colorscheme': 'nucleic'}})
    
    # 4. Выделяем Ионы Цинка (Zinc), которые критически важны для связывания
    view.setStyle({'elem': 'Zn'}, {'sphere': {'color': 'purple', 'radius': 1.5}})
    
    # 5. Фокусируем камеру на месте контакта ДНК и белка
    view.zoomTo()
    
    # Генерируем HTML код
    html_content = f"""
    <html>
    <head>
        <title>CTCF 3D Structure (PDB: {pdb_id})</title>
        <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    </head>
    <body style="margin:0; padding:0; display:flex; justify-content:center; align-items:center; background-color:#f0f0f0;">
        <div style="box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 10px; overflow: hidden;">
            {view._make_html()}
        </div>
    </body>
    </html>
    """
    
    # Сохраняем в файл
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_content)