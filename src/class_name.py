# -*- coding: utf-8 -*-
"""
ImageNet-1k class index -> label mapping (dipakai oleh inference_edit.py).

`inference_edit.py` mengakses `class_name[idx]` di mana idx = argmax(softmax(logits)).
Isi list di bawah dengan 1000 label ImageNet sesuai urutan output model Anda
(mis. ResNet50/MobileNet). Di sini disediakan kerangka + contoh kelas yang
dipakai pada paper (bird/car/train/airplane) agar mudah diadaptasi.

Cara cepat mengisi penuh (opsi):
    import json, urllib.request
    url = "https://raw.githubusercontent.com/raw-imagenet/imagenet-labels/main/imagenet1000.json"
    class_name = list(json.load(urllib.request.urlopen(url)).values())
"""

# TODO: ganti dengan 1000 label ImageNet sesuai urutan output model Anda.
# Placeholder bernomor supaya indexing tetap aman bila belum diisi penuh.
class_name = [f"class_{i}" for i in range(1000)]

# Contoh: petakan beberapa indeks ke label yang dipakai di paper.
# (indeks di bawah hanya ilustrasi — sesuaikan dgn label model Anda)
_examples = {
    # 0: "hummingbird",
    # 1: "airplane",
    # 2: "train",
    # 3: "sports car",
}
for _i, _name in _examples.items():
    if 0 <= _i < len(class_name):
        class_name[_i] = _name
