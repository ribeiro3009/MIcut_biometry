import csv
import os
import subprocess
from pathlib import Path
import time

# Configurações
NFIQ2_PATH = r"bin\NFIQ2\bin\nfiq2.exe"
CROPS_DIR = "crops"
INPUT_CSV = "single.csv"
OUTPUT_CSV = "single.csv"
THREADS = min(os.cpu_count() * 2, 8)
BATCH_FILE = "nfiq2_batch.txt"
TEMP_OUTPUT = "nfiq2_output.csv"  # Manter como .csv

def process_with_batch():
    # 1. Preparar lista de imagens
    with open(INPUT_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Pular cabeçalho
        images_in_csv = {row[0] for row in reader if row}

    # 2. Filtrar imagens existentes
    existing_images = [
        f for f in os.listdir(CROPS_DIR) 
        if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        and f in images_in_csv
    ]

    if not existing_images:
        print("Nenhuma imagem válida encontrada para processamento!")
        return

    # 3. Criar arquivo batch
    with open(BATCH_FILE, 'w') as f:
        for img in existing_images:
            f.write(f"{os.path.abspath(os.path.join(CROPS_DIR, img))}\n")

    # 4. Executar NFIQ2
    print(f"Processando {len(existing_images)} imagens com {THREADS} threads...")
    start_time = time.time()
    
    try:
        subprocess.run(
            [
                NFIQ2_PATH,
                "-f", os.path.abspath(BATCH_FILE),
                "-o", os.path.abspath(TEMP_OUTPUT),
                "-j", str(THREADS),
                "-F"
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar NFIQ2: {e.stderr.decode()}")
        return

    print(f"Processamento concluído em {time.time() - start_time:.2f} segundos")

    # 5. Parsear resultados - VERSÃO AJUSTADA PARA SEU FORMATO
    scores = {}
    try:
        with open(TEMP_OUTPUT, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Remove aspas e caminho completo, mantendo apenas o nome do arquivo
                    filename = Path(row['Filename'].strip('"')).name
                    scores[filename] = row['QualityScore']
                except KeyError:
                    continue
    except Exception as e:
        print(f"Erro ao ler arquivo de saída: {str(e)}")
        return

    # Debug: mostrar alguns scores
    print(f"\nExemplo de scores extraídos (total: {len(scores)}):")
    for filename, score in list(scores.items())[:5]:
        print(f"{filename}: {score}")

    # 6. Atualizar CSV original
    updated_rows = []
    with open(INPUT_CSV, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Adicionar/atualizar coluna de score
        if len(header) < 7:
            header.append("nfiq2_score")
        elif "nfiq2_score" not in header[6].lower():
            header[6] = "nfiq2_score"
        
        updated_rows.append(header)
        
        for row in reader:
            if not row:
                continue
            filename = row[0]
            if filename in scores:
                if len(row) < 7:
                    row.append(scores[filename])
                else:
                    row[6] = scores[filename]
            updated_rows.append(row)

    # 7. Escrever CSV atualizado
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(updated_rows)

    # 8. Limpeza
    for f in [BATCH_FILE, TEMP_OUTPUT]:
        try:
            os.remove(f)
        except:
            pass

    print(f"\nConcluído! {len(scores)} scores adicionados ao arquivo {OUTPUT_CSV}")

if __name__ == "__main__":
    process_with_batch()