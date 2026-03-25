# aotearoa-bird-classifier

clone repo
```
git clone https://github.com/aimeexlin/aotearoa-bird-classifier.git
cd aotearoa-bird-classifier
```

set up env
```
conda env create -f environment.yml python=3.10.9
conda activate species
```

download data files into root
https://drive.google.com/file/d/1L74V_Fqsvj1ku7drcHpBkS2imYLytsZD/view?usp=sharing
https://drive.google.com/file/d/1TXnETXa2do8jMDITqOf4FIBGZME0p1xc/view?usp=sharing
https://drive.google.com/file/d/1eSdpfSNjnh42FFLo3Y4cxZ3025N-YK7a/view?usp=sharing

download data
```
python download_res_grade.py
python download_cap_cul.py
```

clean data
```
python perform_sanitise_instructions.py
```

split into test/train
```
python split.py
```

fine-tune
```
```

validate
```
```