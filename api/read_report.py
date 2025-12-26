
import sys

try:
    # PowerShell > redirection creates UTF-16LE
    with open('confusion_report.txt', 'r', encoding='utf-16le') as f:
        content = f.read()
except:
    try:
        # Fallback
        with open('confusion_report.txt', 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        with open('confusion_report.txt', 'r', errors='ignore') as f:
            content = f.read()

print(content)
