import os
import socket
import urllib.request
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision.datasets.folder import IMG_EXTENSIONS
from multiprocessing.pool import Pool

# decrease wait time for each download
socket.setdefaulttimeout(15)

# IDs and URLs
multimedia = pd.read_csv('multimedia.txt', delimiter='\t')
dataset = pd.read_csv('NZ-Species.csv', delimiter='\t')

# Keep birds only from the research-grade metadata.
birds = dataset.loc[dataset['class'] == 'Aves', ['gbifID', 'verbatimScientificName']].dropna()
birds['gbifID'] = birds['gbifID'].astype(str)
bird_species_by_gbif = dict(zip(birds['gbifID'], birds['verbatimScientificName']))

def download_one(args):
    i, row, species_name = args
    try:
        species_dir = 'res_grade/' + species_name
        filename = species_dir + '/' + str(i) + '_' + str(row['gbifID']) + '.' + str(row['format']).split('/')[-1]
        # skip files in incompatible formats
        if not (str(row['identifier']).lower().endswith(tuple(IMG_EXTENSIONS)) or filename.lower().endswith(tuple(IMG_EXTENSIONS))):
            return f'Skipped {i} {row["gbifID"]} {row["identifier"]}'
        if not os.path.exists(species_dir):
            os.makedirs(species_dir)
        if os.path.exists(filename):
            return None  # Already downloaded
        urllib.request.urlretrieve(row['identifier'], filename)
        try:
            im = Image.open(filename).convert('RGB')
            if filename.lower().endswith(tuple(IMG_EXTENSIONS)):
                im.save(filename)
            else:
                os.remove(filename)
                pil_extension = '.' + im.format.lower()
                if pil_extension in IMG_EXTENSIONS:
                    im.save(filename.split('.')[0] + pil_extension)
                else:
                    return f'Removed incompatible {i} {row["gbifID"]} {row["identifier"]}'
        except Exception:
            os.remove(filename)
            return f'Removed unreadable {i} {row["gbifID"]} {row["identifier"]}'
    except Exception:
        return f'Failed to acquire {i} {row["gbifID"]} {row["identifier"]}'
    return None

if __name__ == '__main__':
    tasks = []
    for i, row in multimedia.iterrows():
        gbif_id = str(row['gbifID'])
        species_name = bird_species_by_gbif.get(gbif_id)
        if species_name is not None:
            tasks.append((i, row, species_name))

    print(f'Bird rows queued: {len(tasks)} / {len(multimedia)}', flush=True)

    pool = Pool(processes=16)  # Adjust number of processes as needed
    results = list(tqdm(pool.imap_unordered(download_one, tasks), total=len(tasks)))
    pool.close()
    pool.join()
    # Print only errors/skipped
    for r in results:
        if r:
            print(r, flush=True)
