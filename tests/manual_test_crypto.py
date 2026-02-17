from enton.skills.crypto_toolkit import CryptoToolkit
import os
import json

def test_crypto():
    print("=== TESTANDO CRYPTO TOOLKIT (Day Trader Doidão) ===")
    
    # Usar carteira de teste
    test_wallet = "/home/gabriel-maia/Documentos/enton/test_wallet.json"
    if os.path.exists(test_wallet):
        os.remove(test_wallet)
        
    toolkit = CryptoToolkit(wallet_path=test_wallet)
    
    # 1. Check Status Inicial
    print("\n--- Status Inicial ---")
    print(toolkit.get_wallet_status())
    
    # 2. Get Price
    print("\n--- Preço Bitcoin ---")
    print(toolkit.get_crypto_price("bitcoin"))
    
    # 3. Sentiment
    print("\n--- Sentimento de Mercado ---")
    print(toolkit.check_market_sentiment())
    
    # 4. Buy Bitcoin
    print("\n--- Comprando $5000 em BTC ---")
    print(toolkit.execute_paper_trade("buy", "bitcoin", 5000.0))
    
    # 5. Status Após Compra
    print("\n--- Status Após Compra ---")
    print(toolkit.get_wallet_status())
    
    # 6. Sell Bitcoin (profit??)
    print("\n--- Vendendo $1000 em BTC ---")
    print(toolkit.execute_paper_trade("sell", "bitcoin", 1000.0))
    
    # 7. Status Final
    print("\n--- Status Final ---")
    print(toolkit.get_wallet_status())

    # Limpeza
    if os.path.exists(test_wallet):
        os.remove(test_wallet)
    print("\n=== TESTE CONCLUÍDO ===")

if __name__ == "__main__":
    test_crypto()
