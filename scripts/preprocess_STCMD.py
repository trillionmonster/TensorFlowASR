import os
import librosa
import glob
from tqdm import tqdm

file_dir = "~/ST-CMDS-20170001_1-OS"
dst_dir = "../tans/ST-CMDS-20170001_1-OS"
vocab_set = set()

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

transcripts = []

text_files = glob.glob(os.path.join(file_dir, "**", "*.txt"), recursive=True)



def write(name, lines):
    with open(os.path.join(dst_dir, name), "w", encoding="utf-8") as out:
        out.write("PATH\tDURATION\tTRANSCRIPT\n")
        for line in tqdm(lines, desc="[Writing]"):
            out.write(line)


for text_file in tqdm(text_files, desc="[Loading]"):
    text = open(text_file).readlines()[0].strip()
    vocab_set = vocab_set.union(set(text))
    audio_file = text_file.replace(".txt", ".wav")

    audio_file = os.path.abspath(audio_file)

    y, sr = librosa.load(audio_file, sr=None)

    duration = librosa.get_duration(y, sr)
    transcripts.append(f"{audio_file}\t{duration:.2f}\t{text}\n")

write("train.tsv", transcripts[0:-20000])
write("dev.tsv", transcripts[-20000:-10000])
write("test.tsv", transcripts[-10000:])
write("all.tsv", transcripts[-10000:])

with open(os.path.join(dst_dir, "vocab.txt"), mode="w") as f:
    vocab = [item + "\n" for item in vocab_set]
    f.writelines(vocab)
