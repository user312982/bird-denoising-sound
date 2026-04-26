import torch

def wav_to_spectrogram(wav, n_fft=1024, hop_length=256, win_length=1024):
    """
    Converts 1D audio waveform to Magnitude and Phase spectrograms.
    
    Args:
        wav: torch.Tensor of shape (B, T) or (B, 1, T)
        n_fft: size of Fourier transform
        hop_length: distance between successive frames
        win_length: window size
        
    Returns:
        magnitude: (B, 1, F, T_frames) - 1-channel image representation
        phase: (B, 1, F, T_frames) - needed for reconstruction
    """
    if wav.dim() == 3:
        wav = wav.squeeze(1) # Ensure shape is (B, T)
        
    window = torch.hann_window(win_length).to(wav.device)
    
    stft_out = torch.stft(
        wav, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        window=window, 
        return_complex=True,
        pad_mode='constant'
    )
    
    magnitude = torch.abs(stft_out)
    phase = torch.angle(stft_out)
    
    # Add channel dimension to treat it as a 1-channel image
    magnitude = magnitude.unsqueeze(1)
    phase = phase.unsqueeze(1)
    
    return magnitude, phase

def spectrogram_to_wav(magnitude, phase, n_fft=1024, hop_length=256, win_length=1024):
    """
    Reconstructs 1D audio waveform from Magnitude and Phase spectrograms.
    
    Args:
        magnitude: (B, 1, F, T_frames) or (B, F, T_frames)
        phase: (B, 1, F, T_frames) or (B, F, T_frames)
        
    Returns:
        wav: torch.Tensor of shape (B, T)
    """
    if magnitude.dim() == 4:
        magnitude = magnitude.squeeze(1)
    if phase.dim() == 4:
        phase = phase.squeeze(1)
        
    # Reconstruct complex STFT representation
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)
    stft_out = torch.complex(real, imag)
    
    window = torch.hann_window(win_length).to(magnitude.device)
    
    wav = torch.istft(
        stft_out, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        window=window
    )
    
    return wav
