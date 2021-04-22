import glob
import librosa
import os.path as osp
import tqdm
import numpy as np
import json

from Parser import Parser
from Tempo import estimateTempo

if __name__ == "__main__":
    parser = Parser()
    sr = 44100
    ps = []
    out = []
    IOI_fix = -1.0 # ms, -1.0 for no fix [20.0, 40.0, 50.0, 70.0, 100.0]
    IOI_factor = -1.0 # portion. ex: 0.2, 0.15.  -1.0 for no factor [0.10, 0.125, 0.15, 0.175, 0.20, 0.225]

    for IOI_fix in [40.0, 70.0]:
        print (f"fix:{IOI_fix}, portion:{IOI_factor}")
        for data in tqdm.tqdm(parser):
            y, T1, T2, ST1, filename = data["y"], data["T1"], data["T2"], data["ST1"], data["filename"]
            tempo1, weight1, tempo2, weight2 = estimateTempo(y, sr, verbose=False, IOI_fix=IOI_fix, IOI_factor=IOI_factor)
            eT1 = min(tempo1, tempo2)
            eT2 = max(tempo1, tempo2)
            #print (filename, T1, T2, ST1, len(y))

            TT1, TT2 = 0.0, 0.0
            if T1 > 0.0 and abs(eT1-T1) / T1 < 0.04:
                TT1 = 1.0
            if T2 > 0.0 and abs(eT2-T2) / T2 < 0.04:
                TT2 = 1.0

            # https://www.music-ir.org/mirex/wiki/2018:Audio_Tempo_Estimation#Evaluation_Procedures
            P = TT1*ST1+TT2*(1-ST1)

            ps.append(P)
            out.append({"Filename":filename, "T1":T1, "eT1":eT1, "T2":T2, "eT2":eT2, "ST1":ST1, "TT1":TT1, "TT2":TT2, "P":P, "tempo1":tempo1, "weigh1":weight1, "tempo2":tempo2, "weight2":weight2})

        avg_p = np.mean(np.array(ps))
        print ("Average P score:{}".format(avg_p))
        with open(f'{IOI_fix}_{IOI_factor}_results.json', 'w') as fp:
            json.dump(out, fp, indent=4)
