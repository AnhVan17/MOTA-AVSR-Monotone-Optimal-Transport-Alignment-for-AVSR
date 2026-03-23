import modal

app = modal.App("list-checkpoints")
volume = modal.Volume.from_name("avsr-checkpoints")

@app.function(volumes={"/checkpoints": volume})
def list_files():
    import os
    from pathlib import Path
    
    print("Listing /checkpoints contents:")
    for root, dirs, files in os.walk("/checkpoints"):
        for name in files:
            print(os.path.join(root, name))
        for name in dirs:
             print(os.path.join(root, name) + "/")

if __name__ == "__main__":
    with app.run():
        list_files.remote()
