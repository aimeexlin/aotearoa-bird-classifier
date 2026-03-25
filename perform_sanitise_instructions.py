import os
import csv
import argparse
from multiprocessing import cpu_count

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from multiprocessing.pool import Pool

# resize data for training
t_512 = transforms.Resize(512)

# move one image for sanitation
def move_single(source, target):
  if os.path.exists(target):
    return
  try:
    im = Image.open(source)
    t_512(im).save(target, format='PNG', optimize=True)
  # log problematic images that cannot be opened or resized
  except:
    print(source, target, flush=True)
  return

def load_bird_class_names():
  bird_classes = set()

  # Research-grade bird taxa from NZ-Species metadata.
  if os.path.exists('NZ-Species.csv'):
    with open('NZ-Species.csv', 'r', newline='') as f:
      reader = csv.DictReader(f, delimiter='\t')
      for row in reader:
        if (row.get('class') or '').strip() == 'Aves':
          name = (row.get('verbatimScientificName') or '').strip()
          if name:
            bird_classes.add(name)

  # Captive/cultivated bird taxa from metadata.
  if os.path.exists('captive_cultivated.csv'):
    with open('captive_cultivated.csv', 'r', newline='') as f:
      reader = csv.DictReader(f)
      for row in reader:
        if (row.get('iconic_taxon_name') or '').strip().lower() == 'aves':
          name = (row.get('scientific_name') or '').strip()
          if name:
            bird_classes.add(name)

  return bird_classes

# move one class for sanitation
def move(source, target, pool):
  if not os.path.exists(f'dataset/{target}/'):
    os.makedirs(f'dataset/{target}/', exist_ok=True)
  # collect images to move into the sanitised class
  move_list = []
  # collect research-grade instances
  if os.path.exists(f'res_grade/{source}/'):
    move_list += [(f'res_grade/{source}/{filename}', f"dataset/{target}/{filename.split('.')[0]}.png") for filename in os.listdir(f'res_grade/{source}/')]
  # collect captive/cultivated instances
  if os.path.exists(f'cap_cul/{source}/'):
    move_list += [(f'cap_cul/{source}/{filename}', f"dataset/{target}/{filename.split('.')[0]}.png") for filename in os.listdir(f'cap_cul/{source}/')]
  # perform the move
  if move_list:
    pool.starmap(move_single, move_list)
  # log problematic empty classes
  else:
    print(source, target, flush=True)
  return

def should_process(parsed_i, birds_only, bird_classes):
  if not birds_only:
    return True

  op = parsed_i[0]
  # For birds-only mode:
  # K target=parsed_i[1]
  # R source=parsed_i[2], target=parsed_i[1]
  # M target=parsed_i[1], sources=parsed_i[2:]
  if op == 'K':
    return parsed_i[1] in bird_classes
  if op == 'R':
    return parsed_i[1] in bird_classes or parsed_i[2] in bird_classes
  if op == 'M':
    if parsed_i[1] in bird_classes:
      return True
    for src in parsed_i[2:]:
      if src in bird_classes:
        return True
  return False


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--instructions', type=str, default='refined_instructions.txt')
  parser.add_argument('--scope', type=str, default='birds', choices=['birds', 'all'])
  parser.add_argument('--workers', type=int, default=min(32, cpu_count() or 8))
  args = parser.parse_args()

  birds_only = args.scope == 'birds'
  bird_classes = load_bird_class_names() if birds_only else set()

  if birds_only:
    print(f'Bird classes loaded: {len(bird_classes)}', flush=True)

  with open(args.instructions, 'r') as f:
    instructions = f.read().split('\n')

  with Pool(processes=args.workers) as pool:
    for i in tqdm(instructions):
      if ',' not in i:
        continue
      parsed_i = i.split(',')
      if not should_process(parsed_i, birds_only, bird_classes):
        continue

      # delete
      if parsed_i[0] == 'D':
        continue
      # keep
      if parsed_i[0] == 'K':
        move(parsed_i[1], parsed_i[1], pool)
      # rename
      elif parsed_i[0] == 'R':
        move(parsed_i[2], parsed_i[1], pool)
      # merge
      elif parsed_i[0] == 'M':
        for p_i in parsed_i[2:]:
          move(p_i, parsed_i[1], pool)


if __name__ == '__main__':
  main()
