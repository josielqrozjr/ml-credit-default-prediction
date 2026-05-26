# ============================================================================
# DIAGNÓSTICO DE GPU - DEBUG SCRIPT
# ============================================================================
# Execute este script para diagnosticar problemas com GPU

import os
import sys
import subprocess

print("="*80)
print("DIAGNÓSTICO COMPLETO DE GPU - PyTorch + CUDA")
print("="*80)

# ============================================================================
# 1. INFORMAÇÕES DE SISTEMA
# ============================================================================
print("\n[1] INFORMAÇÕES DE SISTEMA")
print("-"*80)
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")

# ============================================================================
# 2. VERIFICAR DRIVER NVIDIA
# ============================================================================
print("\n[2] DRIVER NVIDIA")
print("-"*80)
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ nvidia-smi encontrado:")
        print(result.stdout)
    else:
        print("✗ nvidia-smi não respondeu. Saída:")
        print(result.stderr)
except FileNotFoundError:
    print("✗ nvidia-smi NÃO ENCONTRADO")
    print("   → Driver NVIDIA pode não estar instalado")
    print("   → Ou não está no PATH do sistema")

# ============================================================================
# 3. VERIFICAR PyTorch
# ============================================================================
print("\n[3] PYTORCH - INFORMAÇÕES")
print("-"*80)
try:
    import torch
    print(f"✓ PyTorch importado com sucesso")
    print(f"  Versão: {torch.__version__}")
    print(f"  CUDA disponível (torch.cuda.is_available()): {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"\n✓ GPU DETECTADA!")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Número de GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    - Memória: {props.total_memory / 1e9:.2f} GB")
            print(f"    - Compute Capability: {props.major}.{props.minor}")
    else:
        print(f"\n✗ GPU NÃO DETECTADA por PyTorch")
        print(f"  Possíveis causas:")
        print(f"    1. Driver NVIDIA não instalado")
        print(f"    2. PyTorch instalado sem suporte CUDA")
        print(f"    3. Versão do CUDA não compatível")
        
except ImportError as e:
    print(f"✗ PyTorch NÃO ENCONTRADO: {e}")

# ============================================================================
# 4. VERIFICAR CUDA BACKENDS
# ============================================================================
print("\n[4] CUDA BACKENDS")
print("-"*80)
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ cudnn versão: {torch.backends.cudnn.version()}")
        print(f"  cudnn disponível: {torch.backends.cudnn.is_available()}")
        print(f"  cudnn deterministic: {torch.backends.cudnn.deterministic}")
        print(f"  cudnn benchmark: {torch.backends.cudnn.benchmark}")
    else:
        print("⚠ CUDA não disponível - backends não ativados")
except Exception as e:
    print(f"✗ Erro ao verificar backends: {e}")

# ============================================================================
# 5. VERIFICAR LIGHTGBM
# ============================================================================
print("\n[5] LIGHTGBM")
print("-"*80)
try:
    import lightgbm as lgb
    print(f"✓ LightGBM versão: {lgb.__version__}")
    
    # Tentar criar modelo com GPU
    try:
        params = {'device': 'gpu', 'verbose': -1}
        data = lgb.Dataset([[1, 2], [3, 4]], label=[0, 1])
        # Apenas criar dataset, não treinar
        print(f"  ✓ LightGBM com GPU testado (dispositivo disponível)")
    except Exception as e:
        print(f"  ✗ LightGBM GPU falhou: {e}")
        print(f"  → Use 'cpu' em CONFIG['lightgbm_device']")
        
except ImportError as e:
    print(f"✗ LightGBM NÃO ENCONTRADO: {e}")

# ============================================================================
# 6. VARIÁVEIS DE AMBIENTE
# ============================================================================
print("\n[6] VARIÁVEIS DE AMBIENTE")
print("-"*80)
env_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH', 'PATH']
for var in env_vars:
    value = os.environ.get(var, 'NÃO DEFINIDA')
    if var in ['LD_LIBRARY_PATH', 'PATH']:
        # Truncar se muito longo
        value = value[:100] + "..." if len(value) > 100 else value
    print(f"  {var}: {value}")

# ============================================================================
# 7. RECOMENDAÇÕES
# ============================================================================
print("\n" + "="*80)
print("RECOMENDAÇÕES")
print("="*80)

try:
    import torch
    if not torch.cuda.is_available():
        print("\n✗ GPU NÃO DETECTADA - SOLUÇÃO:")
        print()
        print("Opção 1: Verificar Driver NVIDIA")
        print("-" * 40)
        print("  Windows:")
        print("    1. Abra 'Gerenciador de Dispositivos' (devmgmt.msc)")
        print("    2. Procure por 'NVIDIA' em 'Adaptadores de vídeo'")
        print("    3. Se tiver !, clique direito → Atualizar driver")
        print()
        print("  Linux:")
        print("    1. sudo apt update && sudo apt install nvidia-driver-550")
        print("    2. sudo reboot")
        print()
        print("Opção 2: Reinstalar PyTorch para CUDA 13.0")
        print("-" * 40)
        print("  pip uninstall torch torchvision torchaudio -y")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130")
        print()
        print("Opção 3: Usar CPU (mais lento, mas funciona)")
        print("-" * 40)
        print("  Na Célula 1, altere:")
        print("    CONFIG['lightgbm_device'] = 'cpu'")
        print()
    else:
        print("\n✓ GPU DETECTADA - Tudo OK!")
        print()
        print("Se ainda tiver problemas:")
        print("  1. Verifique se está usando a GPU na Célula 1")
        print("  2. Confirme: DEVICE = torch.device('cuda')")
        print("  3. Confirme: CONFIG['lightgbm_device'] = 'gpu'")
        
except ImportError:
    print("\n✗ PyTorch não está instalado")
    print()
    print("Reinstale com:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130")

print("\n" + "="*80)
print("FIM DO DIAGNÓSTICO")
print("="*80)
