import glob

datadir = 'partim/'

parfiles = sorted(glob.glob(datadir + '*.par'))

for parfile in parfiles:
    newfile = parfile.replace("NANOGrav_12yv4.gls", "dmxset")
    with open(newfile, 'w') as fnew:
        with open(parfile, 'r') as fold:
            for line in fold:
                line_split = line.split()
                if 'DMX_' in line_split[0]:
                    line_split[2] = str(0)
                    new_line = "\t".join(line_split) + "\n"
                    fnew.write(new_line)
                else:
                    fnew.write(line)
    print(f"Modified par file {newfile} written!")
