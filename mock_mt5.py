# mock_mt5.py

class MockOrderSendResult:
    def __init__(self):
        self.retcode = 10009
        self.order = 123456
        self.comment = "Simulated order executed"
        self.request_id = 999

class MockMT5:
    def OrderSend(self, request):
        print(f"[MOCK] OrderSend called: {request}")
        return MockOrderSendResult()

    def initialize(self, *args, **kwargs):
        print("[MOCK] MT5 initialized.")
        return True

    def shutdown(self):
        print("[MOCK] MT5 shutdown.")

    def terminal_info(self):
        return {"name": "MockTerminal", "version": "1.0"}

    def account_info(self):
        return {"balance": 100_000, "equity": 100_000}

OrderSendResult = MockOrderSendResult
