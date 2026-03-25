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
multimedia = pd.read_csv('captive_cultivated.csv', delimiter=',')

# Keep birds only from captive/cultivated metadata.
bird_multimedia = multimedia[
    multimedia['iconic_taxon_name'].fillna('').str.lower().eq('aves')
]

def download_one(args):
    i, row = args
    try:
        if row['license'] not in ['CC-BY', 'CC-BY-NC', 'CC0']:
            return None
        species_dir = 'cap_cul/' + row['scientific_name']
        filename = species_dir + '/cap_cul_' + str(i) + '_' + str(row['id']) + '.' + str(row['image_url']).split('.')[-1]
        # skip files in incompatible formats
        if not filename.lower().endswith(tuple(IMG_EXTENSIONS)):
            return f'Skipped {i} {row["id"]} {row["image_url"]}'
        if not os.path.exists(species_dir):
            os.makedirs(species_dir)
        if os.path.exists(filename):
            return None  # Already downloaded
        url = row['image_url'].replace('small', 'original').replace('medium', 'original').replace('large', 'original')
        urllib.request.urlretrieve(url, filename)
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
                    return f'Removed incompatible {i} {row["id"]} {row["image_url"]}'
        except Exception:
            os.remove(filename)
            return f'Removed unreadable {i} {row["id"]} {row["image_url"]}'
    except Exception:
        return f'Failed to acquire {i} {row["id"]} {row["image_url"]}'
    return None

if __name__ == '__main__':
    tasks = list(bird_multimedia.iterrows())
    print(f'Bird rows queued: {len(tasks)} / {len(multimedia)}', flush=True)

    pool = Pool(processes=16)  # Adjust number of processes as needed
    results = list(tqdm(pool.imap_unordered(download_one, tasks), total=len(tasks)))
    pool.close()
    pool.join()
    # Print only errors/skipped
    for r in results:
        if r:
            print(r, flush=True)
