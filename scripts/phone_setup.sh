#!/data/data/com.termux/files/usr/bin/bash
# =============================================================================
# Enton Phone Setup — roda DENTRO do Termux no celular
#
# Pre-requisitos:
#   1. Instalar Termux do F-Droid (NAO da Play Store!)
#   2. Instalar Termux:API do F-Droid
#   3. Abrir Termux e colar este script
#
# Uso:
#   curl -sL <url> | bash
#   ou copiar e colar no Termux
# =============================================================================

set -e

echo "=== Enton Phone Setup ==="
echo ""

# 1. Atualizar pacotes
echo "[1/4] Atualizando pacotes..."
pkg update -y
pkg upgrade -y

# 2. Instalar dependencias
echo "[2/4] Instalando pacotes..."
pkg install -y termux-api openssh python

# 3. Configurar SSH
echo "[3/4] Configurando SSH..."
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Definir senha do Termux (necessario pro SSH)
echo ""
echo ">>> Defina uma senha pro Termux (usada pra SSH):"
passwd

# Iniciar SSH server
sshd
echo "SSH server iniciado na porta 8022"

# 4. Mostrar info de conexao
echo ""
echo "[4/4] Setup completo!"
echo ""
echo "============================================"
echo "  Info de conexao"
echo "============================================"
IP=$(ifconfig wlan0 2>/dev/null | grep 'inet ' | awk '{print $2}' || echo "N/A")
echo "  IP WiFi:  $IP"
echo "  SSH:      ssh -p 8022 $(whoami)@$IP"
echo "  Termux:   $(whoami)"
echo ""
echo "  Do PC (copiar chave SSH):"
echo "    ssh-copy-id -p 8022 $(whoami)@$IP"
echo ""
echo "  Testar Termux API:"
echo "    termux-battery-status"
echo "    termux-clipboard-get"
echo "    termux-location -p gps"
echo "============================================"
echo ""
echo "Pronto! O Enton agora pode acessar o celular via SSH + Termux API."
echo "Pra manter o SSH rodando, deixe o Termux aberto ou use Termux:Boot."
