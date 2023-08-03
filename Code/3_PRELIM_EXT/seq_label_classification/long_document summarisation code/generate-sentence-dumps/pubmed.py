import os
from shutil import copyfile

OUTDIR = './out'
SRCDIR = './src'
FOLDERS = ['train_processed', 'test_processed', 'val_processed']

if __name__=='__main__':
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
    count = 0
    for folder in FOLDERS:
        entries = os.listdir(os.path.join(SRCDIR, folder))
        uniqueNames = set(map(lambda x: x.split(".")[0], entries))
        for name in uniqueNames:
            copyfile(os.path.join(SRCDIR, folder, name+'.src'), os.path.join(OUTDIR, str(count)+".src"))
            copyfile(os.path.join(SRCDIR, folder, name+'.tgt'), os.path.join(OUTDIR, str(count)+".tgt"))
            count+=1


