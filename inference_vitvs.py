import os
import torch
import torchaudio
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

from models.vitvs import ViTVS_Segmenter
from models.audio_processing import wav_to_spectrogram, spectrogram_to_wav

def pad_or_trim(tensor, target_length):
    """Pad or trim a 1D tensor to the target length"""
    if tensor.shape[-1] > target_length:
        return tensor[..., :target_length]
    elif tensor.shape[-1] < target_length:
        pad_size = target_length - tensor.shape[-1]
        return F.pad(tensor, (0, pad_size))
    return tensor

def denoise_audio(
    noisy_wav_path, 
    model_path, 
    output_path="denoised_output.wav",
    device="cuda",
    target_sample_rate=16000,
    img_size=(256, 256),
    threshold=0.5
):
    """
    ViTVS Pipeline Inference:
    Audio -> STFT -> Resize 256x256x3 -> ViTVS Segmenter -> Mask -> Restore Size -> Apply S'[Mask<1]=0 -> ISTFT
    """
    print(f"1. Memuat audio: {noisy_wav_path}")
    wav, sr = torchaudio.load(noisy_wav_path)
    
    # Resample jika diperlukan
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        wav = resampler(wav)
    
    # Pastikan mono (1 channel)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        
    wav = wav.to(device)
    
    print("2. STFT: Konversi audio ke Spectrogram (Magnitude & Phase)")
    magnitude, phase = wav_to_spectrogram(wav, n_fft=1024, hop_length=256, win_length=1024)
    
    orig_freq_bins = magnitude.shape[2]
    orig_time_frames = magnitude.shape[3]
    
    print(f"   Ukuran spectrogram asli: Freq={orig_freq_bins}, Time={orig_time_frames}")
    
    print("3. Preprocessing gambar untuk ViTVS")
    # Normalize magnitude ke range [0, 1]
    mag_min, mag_max = magnitude.min(), magnitude.max()
    img_tensor = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)
    
    # Resize ke 256x256
    img_resized = F.interpolate(img_tensor, size=img_size, mode='bilinear', align_corners=False)
    
    # Duplikasi 1 channel jadi 3 channel (RGB) untuk input ViT
    img_rgb = img_resized.repeat(1, 3, 1, 1)
    
    # Normalize dengan ImageNet stats (sesuai dataset loader)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_input = normalize(img_rgb[0]).unsqueeze(0)  # (1, 3, 256, 256)
    
    print(f"4. Memuat model ViTVS dari {model_path}")
    model = ViTVS_Segmenter(
        img_size=img_size, 
        patch_size=16, 
        in_chans=3, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        num_classes=2
    ).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("   Model berhasil dimuat!")
    else:
        print(f"   [PERINGATAN] File model '{model_path}' tidak ditemukan. Menggunakan model baru (belum dilatih)!")
        
    model.eval()
    with torch.no_grad():
        logits = model(img_input)
        probs = torch.softmax(logits, dim=1)
        
        # Probabilitas kelas 1 (Clean Audio)
        clean_prob = probs[:, 1:2, :, :]  
        
        # Binary mask
        mask_resized = (clean_prob > threshold).float()
    
    print("5. Post-processing: Menerapkan Mask ke Spectrogram aslinya")
    mask_original_size = F.interpolate(mask_resized, size=(orig_freq_bins, orig_time_frames), mode='nearest')
    
    # Filter noise di frekuensi (magnitude * mask)
    filtered_magnitude = magnitude * mask_original_size
    
    print("6. ISTFT: Rekonstruksi bentuk Audio")
    denoised_wav = spectrogram_to_wav(filtered_magnitude, phase, n_fft=1024, hop_length=256, win_length=1024)
    denoised_wav = pad_or_trim(denoised_wav, wav.shape[-1])
    
    print(f"7. Menyimpan hasil audio ke {output_path}")
    torchaudio.save(output_path, denoised_wav.cpu(), target_sample_rate)
    print("Selesai!")
    
    return wav, denoised_wav, magnitude, mask_original_size, filtered_magnitude

if __name__ == "__main__":
    import sys
    
    MODEL_PATH = "best_vitvs_segmenter.pth"
    INPUT_AUDIO = "test_audio.wav"
    OUTPUT_AUDIO = "denoised_test_audio_vitvs.wav"
    
    if not os.path.exists(INPUT_AUDIO):
        print(f"Membuat file audio dummy '{INPUT_AUDIO}'...")
        dummy_wav = torch.randn(1, 16000 * 3) # 3 detik
        torchaudio.save(INPUT_AUDIO, dummy_wav, 16000)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    denoise_audio(
        noisy_wav_path=INPUT_AUDIO,
        model_path=MODEL_PATH,
        output_path=OUTPUT_AUDIO,
        device=device
    )
