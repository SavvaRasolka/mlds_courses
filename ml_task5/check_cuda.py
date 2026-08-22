import torch

# Проверяем, доступна ли CUDA (технология NVIDIA для работы с GPU)
cuda_available = torch.cuda.is_available()
print(f"CUDA доступна: {cuda_available}")

if cuda_available:
    # Выводим информацию о первой найденной видеокарте
    print(f"Количество GPU: {torch.cuda.device_count()}")
    print(f"Имя GPU: {torch.cuda.get_device_name(0)}")
    print(f"Версия CUDA, с которой собран PyTorch: {torch.version.cuda}")
else:
    print("PyTorch не видит GPU. Будут использоваться только CPU.")