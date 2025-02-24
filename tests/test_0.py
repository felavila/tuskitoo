from SHEAP.module_a import function_to_test

def test_function_to_test():
    assert function_to_test(2) == 4
    assert function_to_test(-3) == -6
    assert function_to_test(0) == 0