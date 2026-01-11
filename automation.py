import subprocess
from main import main
from config import Config

if __name__ == "__main__":
    cfg = Config()
    mpr = cfg.get_model_params()

    base_size = mpr['model_params']['codebook_size']
    base_dim  = mpr['model_params']['codebook_dim']

    for i in range(4):
        current_size = base_size * (2**(i))
        mpr['model_params']['codebook_size'] = current_size
        
        for j in range(6):
            current_dim = base_dim * (2**(j))
            mpr['model_params']['codebook_dim'] = current_dim
            
            print(f"   Round {i+1}: Codebook Size {current_size}")
            print(f"   Running with Dim: {current_dim}")

            try:
                main(cfg, mpr)
            except Exception as e:
                print(f"   !!! Error at Size {current_size}, Dim {current_dim}: {e}")
                
        print("\n")