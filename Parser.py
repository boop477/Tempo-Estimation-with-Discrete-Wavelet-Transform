import glob
import librosa
import os.path as osp
import tqdm
import numpy as np

class Parser(object):
    def __init__(self, audio_path="./giantsteps-tempo-dataset/audio", 
                       annotation_mirex_path="./giantsteps-tempo-dataset/annotations_v2/mirex",
                       sr = 44100):
        audio_paths = [p for p in sorted(glob.glob(osp.join(audio_path, "*.wav")))]
        audio_paths = [p for p in sorted(glob.glob(osp.join(audio_path, "*.wav")))]
        #annotation_paths = [p for p in sorted(glob.glob(osp.join(annotation_mirex_path, "*.mirex")))]

        self.data = []
        self.sr = sr
        for audio_path in tqdm.tqdm(audio_paths):
            filename = audio_path.split("/")[-1].replace(".wav", "")
            annotation_path = osp.join(annotation_mirex_path, audio_path.split("/")[-1].replace(".wav", ".mirex"))
            if osp.isfile(annotation_path) == False:
                print (f"Skip {audio_path} | {annotation_path}")
                continue

            # 1. load audio
            #y, sr = librosa.load(audio_path, sr=self.sr)
            #np.save(audio_path, y)
            y = np.load(audio_path+".npy")

            # 2. load annotation
            with open (annotation_path, "r") as fp:
                annotation = [l for l in fp.readlines()]
            T1, T2, ST1 = annotation[0].split("\t")[0], annotation[0].split("\t")[1], annotation[0].split("\t")[2]

            _data = {"y":y, "T1":float(T1), "T2":float(T2), "ST1":float(ST1), "filename":filename}
            self.data.append(_data)

    def __getitem__(self, key):
        if key < len(self.data):
            return self.data[key]
        else:
            raise StopIteration

if __name__ == "__main__":
    parser = Parser()

    for data in parser:
        y, T1, T2, ST1, filename = data["y"], data["T1"], data["T2"], data["ST1"], data["filename"]
        print (filename, T1, T2, ST1, len(y))