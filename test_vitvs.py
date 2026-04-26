import torch
import torch.nn as nn
from models.vitvs import ViTVS

def test_vitvs():
    # 1. Create a dummy audio waveform (Batch=2, Length=16000)
    # 16000 samples @ 16kHz = 1 second of audio
    batch_size = 2
    audio_length = 16000
    wav = torch.randn(batch_size, audio_length)
    
    print(f"Input Audio Shape: {wav.shape}")
    
    # 2. Initialize the ViTVS model
    # We use a smaller model configuration for fast testing
    model = ViTVS(
        img_size=(513, 63), # Approximate size for n_fft=1024, hop=256, 16000 samples
        patch_size=16,
        embed_dim=256,   
        depth=4,         
        num_heads=4,
        num_classes=2,
        n_fft=1024,
        hop_length=256,
        win_length=1024
    )
    
    print(f"Initialized ViTVS Model. Parameter count: {sum(p.numel() for p in model.parameters())}")
    
    # 3. Run forward pass
    denoised_wav, logits = model(wav)
    
    print(f"Output Audio Shape: {denoised_wav.shape}")
    print(f"Output Logits Shape: {logits.shape}")
    
    # 4. Assertions to verify shapes
    assert denoised_wav.shape == wav.shape, f"Shape mismatch: {denoised_wav.shape} vs {wav.shape}"
    
    # 5. Test Loss Calculation
    # Dummy labels (0 or 1) for each pixel in the spectrogram
    labels = torch.randint(0, 2, logits.shape[2:]).unsqueeze(0).expand(batch_size, -1, -1)
    
    # NLL Loss expects LogSoftmax inputs
    log_probs = nn.functional.log_softmax(logits, dim=1)
    loss_fn = nn.NLLLoss()
    loss = loss_fn(log_probs, labels)
    
    print(f"Successfully calculated NLL Loss: {loss.item():.4f}")
    print("All tests passed!")

if __name__ == "__main__":
    test_vitvs()
