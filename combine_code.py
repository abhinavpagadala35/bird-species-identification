import os
code = ''
for root, _, files in os.walk('.'):
    for f in files:
        if f.endswith(('.py', '.html', '.css', '.js')) and 'combine_code.py' not in f and 'combined_code' not in f and 'venv' not in root and '.gemini' not in root:
            try:
                with open(os.path.join(root, f), 'r', encoding='utf-8') as file:
                    ext = f.split('.')[-1]
                    code += f'\n\n### {f}\n```\n{ext}\n{file.read()}\n```\n'
            except Exception as e:
                print(f"Error reading {f}: {e}")

with open('combined_code.md', 'w', encoding='utf-8') as o:
    o.write(code)
print("done")
