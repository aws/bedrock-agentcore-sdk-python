import os

def test_pwn():
    print("PWN_TEST:", bool(os.getenv("AWS_ACCESS_KEY_ID")))
