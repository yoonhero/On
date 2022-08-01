import json
import torch
import openpyxl
import time

# Load Json File


def load_json(filename: str) -> dict:
    with open(filename, "r") as f:
        return json.load(f)


# Get Device Training Environment
def get_device(isMac: bool = False) -> str:
    if isMac:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Read Excel(.xlsx) File
#         # 셀 주소로 값 출력
#         print(load_ws['B2'].value)
#         # 셀 좌표로 값 출력
#         print(load_ws.cell(3, 2).value)
def load_xlsx(filename: str, sheet_name: str = "Sheet1"):
    wb = openpyxl.load_workbook(filename, data_only=True)
    sheet = wb[sheet_name]

    all_values = [s.value for s in sheet.rows]

    return all_values


# Logging Time Package Class
class TimeLogger:
    def __init__(self, func):
        def logger(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print(f"Calling {func.__name__}: {time.time() - start:.5f}s")
            return result
        self._logger = logger

    def __call__(self, *args, **kwargs):
        return self._logger(*args, **kwargs)


if __name__ == '__main__':
    @TimeLogger
    def calculate_sum_n_cls(n):
        return sum(range(n))

    calculate_sum_n_cls(200000)
