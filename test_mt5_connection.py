import os
import MetaTrader5 as mt5
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MT5_LOGIN = int(os.getenv("MT5_LOGIN", ""))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")

def test_mt5_connection():
    """ Test connection to MT5 """
    if mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print("✅ MT5 Connected Successfully!")
        print("Account Info:", mt5.account_info())
        mt5.shutdown()
    else:
        print(f"❌ MT5 Connection Failed! Error: {mt5.last_error()}")

if __name__ == "__main__":
    test_mt5_connection()
