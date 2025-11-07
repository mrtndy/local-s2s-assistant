import platform
import ctranslate2 as c
import faster_whisper as fw

print("python", platform.python_version())
print("os", platform.system(), platform.release())
print("faster-whisper", fw.__version__)
print("ctranslate2", c.__version__)
print("cpu compute types:", c.get_supported_compute_types("cpu"))
print("cuda compute types:", c.get_supported_compute_types("cuda"))
