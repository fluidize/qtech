import asyncio
import sys
import os

# Add the jupiter directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from jupiter import JupiterWalletHandler, Token

async def test_get_order_async():
    """Test the async get_order method"""
    print("Testing get_order_async method...")
    
    # Use the same private key as in the main script
    private_key = "2BmZhw6gq2VyyvQNhzbXSPp1riXVDQqfiBNPeALf54gsZ9Wh4bLzQrzbysRUgxZVmi862VcXTwFvcAnfC1KYwWsz"
    jupiter = JupiterWalletHandler(private_key)
    
    print(f"Wallet public key: {jupiter.wallet.pubkey()}")
    
    # Test multiple async calls
    test_amounts = [0.1, 0.5, 1.0, 2.0]
    
    for i, amount in enumerate(test_amounts):
        print(f"\n--- Test {i+1}: {amount} SOL to USDC ---")
        
        try:
            result = await jupiter.get_order_async(Token.SOL, Token.USDC, amount)
            
            if result:
                in_usd, out_usd, slippage_bps, fee_bps, price_impact_pct, price_impact_usd, unsigned_tx = result
                print(f"✅ SUCCESS: {amount} SOL → ${out_usd:.2f} USDC")
                print(f"   In USD: ${in_usd:.2f}")
                print(f"   Out USD: ${out_usd:.2f}")
                print(f"   Slippage: {slippage_bps:.1f} bps")
                print(f"   Fee: {fee_bps:.1f} bps")
                print(f"   Price Impact: {price_impact_pct:.4f}%")
                print(f"   Price Impact USD: ${price_impact_usd:.2f}")
                print(f"   TX Length: {len(unsigned_tx)} chars")
            else:
                print(f"❌ FAILED: Could not get order for {amount} SOL")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    # Test concurrent calls
    print(f"\n--- Testing concurrent calls ---")
    tasks = []
    for i in range(5):
        task = jupiter.get_order_async(Token.SOL, Token.USDC, 0.1)
        tasks.append(task)
    
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Concurrent test {i+1} failed: {result}")
            elif result:
                in_usd, out_usd, slippage_bps, fee_bps, price_impact_pct, price_impact_usd, unsigned_tx = result
                print(f"✅ Concurrent test {i+1}: ${out_usd:.2f} USDC")
            else:
                print(f"❌ Concurrent test {i+1}: No result")
    except Exception as e:
        print(f"❌ Concurrent test failed: {e}")

async def test_websocket_simulation():
    """Simulate the websocket loop environment"""
    print(f"\n--- Simulating websocket loop environment ---")
    
    private_key = "2BmZhw6gq2VyyvQNhzbXSPp1riXVDQqfiBNPeALf54gsZ9Wh4bLzQrzbysRUgxZVmi862VcXTwFvcAnfC1KYwWsz"
    jupiter = JupiterWalletHandler(private_key)
    
    # Simulate rapid calls like in a websocket loop
    for i in range(10):
        print(f"Websocket simulation call {i+1}/10...")
        
        try:
            result = await jupiter.get_order_async(Token.SOL, Token.USDC, 0.1)
            if result:
                in_usd, out_usd, slippage_bps, fee_bps, price_impact_pct, price_impact_usd, unsigned_tx = result
                print(f"  ✅ Got quote: ${out_usd:.2f} USDC")
            else:
                print(f"  ❌ No quote received")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # Small delay to simulate real websocket timing
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    print("Starting Jupiter async tests...")
    
    # Run the tests
    asyncio.run(test_get_order_async())
    asyncio.run(test_websocket_simulation())
    
    print("\n✅ All tests completed!") 