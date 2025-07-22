import csv
import os
import subprocess
from pathlib import Path
import time
import multiprocessing
from collections import defaultdict

# Configurações de Performance
NFIQ2_PATH = r"bin\NFIQ2\bin\nfiq2.exe"
CROPS_DIR = "crops"
INPUT_CSV = "single.csv"
OUTPUT_CSV = "single_processed.csv"  # Novo arquivo para evitar sobrescrita
THREADS = max(multiprocessing.cpu_count() - 1, 1)  # Deixa 1 core livre
BATCH_SIZE = 50000  # Processa em lotes para evitar memory overflow
TEMP_DIR = "temp_nfiq2"

def setup_environment():
    """Prepara diretórios temporários"""
    os.makedirs(TEMP_DIR, exist_ok=True)

def process_batch(batch_files, batch_num):
    """Processa um lote de arquivos"""
    batch_path = os.path.join(TEMP_DIR, f"batch_{batch_num}.txt")
    output_path = os.path.join(TEMP_DIR, f"output_{batch_num}.csv")
    
    # Criar arquivo batch
    with open(batch_path, 'w', encoding='utf-8') as f:
        f.writelines(f"{os.path.abspath(os.path.join(CROPS_DIR, img))}\n" for img in batch_files)
    
    # Executar NFIQ2 com prioridade alta
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    
    try:
        subprocess.run(
            [
                NFIQ2_PATH,
                "-f", os.path.abspath(batch_path),
                "-o", os.path.abspath(output_path),
                "-j", str(THREADS),
                "-F",
                "-q"  # Modo quieto para melhor performance
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo
        )
    except subprocess.CalledProcessError as e:
        print(f"Erro no lote {batch_num}: {e.stderr.decode('utf-8', errors='ignore')}")
        return None
    
    return output_path

def parse_output(output_path):
    """Parse ultra-rápido do arquivo de saída"""
    scores = {}
    try:
        with open(output_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    filename = Path(row['Filename'].strip('"')).name
                    scores[filename] = row['QualityScore']
                except (KeyError, AttributeError):
                    continue
    except Exception as e:
        print(f"Erro ao parsear {output_path}: {str(e)}")
    return scores

def process_large_scale():
    setup_environment()
    start_time = time.time()
    
    # 1. Ler apenas os nomes de arquivo do CSV de forma eficiente
    print("[1/5] Lendo arquivo CSV...")
    with open(INPUT_CSV, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Pular cabeçalho
        images_in_csv = {row[0] for row in reader if row}
    
    # 2. Listar imagens disponíveis de forma otimizada
    print("[2/5] Listando imagens...")
    existing_images = []
    for entry in os.scandir(CROPS_DIR):
        if entry.name in images_in_csv and entry.name.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            existing_images.append(entry.name)
    
    total_images = len(existing_images)
    if not total_images:
        print("Nenhuma imagem válida encontrada!")
        return
    
    print(f"Total de imagens para processar: {total_images:,}")
    
    # 3. Processar em lotes
    print("[3/5] Processando lotes...")
    scores = {}
    processed = 0
    batch_num = 1
    
    for i in range(0, total_images, BATCH_SIZE):
        batch = existing_images[i:i + BATCH_SIZE]
        output_file = process_batch(batch, batch_num)
        
        if output_file:
            batch_scores = parse_output(output_file)
            scores.update(batch_scores)
            processed += len(batch_scores)
            print(f"Lote {batch_num}: {len(batch_scores):,} scores | Total: {processed:,}/{total_images:,}")
        
        batch_num += 1
        # Limpeza incremental para economizar memória
        temp_files = [f for f in os.listdir(TEMP_DIR) if f.startswith(f"batch_{batch_num-1}") or f.startswith(f"output_{batch_num-1}")]
        for f in temp_files:
            try:
                os.remove(os.path.join(TEMP_DIR, f))
            except:
                pass
    
    # 4. Atualizar CSV de forma otimizada
    print("[4/5] Atualizando CSV...")
    temp_output = OUTPUT_CSV + ".tmp"
    
    with open(INPUT_CSV, 'r', newline='', encoding='utf-8') as fin, \
         open(temp_output, 'w', newline='', encoding='utf-8') as fout:
        
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        
        # Processar cabeçalho
        header = next(reader)
        if len(header) < 7:
            header.append("nfiq2_score")
        elif "nfiq2_score" not in header[6].lower():
            header[6] = "nfiq2_score"
        writer.writerow(header)
        
        # Processar linhas
        for row in reader:
            if not row:
                continue
            filename = row[0]
            if filename in scores:
                if len(row) < 7:
                    row.append(scores[filename])
                else:
                    row[6] = scores[filename]
            writer.writerow(row)
    
    # 5. Finalização
    print("[5/5] Finalizando...")
    os.replace(temp_output, OUTPUT_CSV)
    
    # Limpeza final
    try:
        os.rmdir(TEMP_DIR)
    except:
        pass
    
    total_time = time.time() - start_time
    print(f"\nProcessamento concluído!")
    print(f"Total de imagens: {total_images:,}")
    print(f"Total processado: {processed:,}")
    print(f"Tempo total: {total_time:.2f} segundos")
    print(f"Velocidade: {total_images/total_time:.2f} img/segundo")
    print(f"Arquivo de saída: {OUTPUT_CSV}")

if __name__ == "__main__":
    process_large_scale()