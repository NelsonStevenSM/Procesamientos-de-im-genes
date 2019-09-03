import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
lista = ["P8100", "P23500", "P71000"]

[os.makedirs(i,exist_ok=True) for i in lista]


for i in lista:
    os.system("cp {}.h264 ./{}".format(i,i))

    image_dir = os.path.join(BASE_DIR, i)
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("h264"):
                path = os.path.join(root, file)

                shell = "ffmpeg -i {} -vf \"select='eq(pict_type, PICT_TYPE_I)'\" -vsync vfr -filter:v fps=fps={}/1 img_%4d.tiff -hide_banner".format(path,25)
                os.system(shell)
                os.system("mv *.tiff ./{}".format(i))
                os.system("rm ./{}/*.h264".format(i))


