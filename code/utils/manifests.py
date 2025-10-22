import yaml, sys, glob

def validate(manifest_glob):
    ok = True
    for fn in glob.glob(manifest_glob):
        with open(fn, 'r', encoding='utf-8') as f:
            y = yaml.safe_load(f)
        if 'items' not in y or not isinstance(y['items'], list):
            print(f"[ERR] {fn}: missing items list"); ok = False
        for i, it in enumerate(y.get('items', [])):
            if 'id' not in it or 'path' not in it:
                print(f"[ERR] {fn} item {i}: missing id/path"); ok = False
    print("Validation:", "OK" if ok else "FAILED")
    return 0 if ok else 2

if __name__ == "__main__":
    sys.exit(validate(sys.argv[sys.argv.index('--validate')+1]))
