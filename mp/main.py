import os
from multiprocessing import Process, set_start_method

def foo(i):
    print(f"Process {os.getpid()} in {i}...")
    print(f"Exiting {i}...")

if __name__ == "__main__":
    print(f"Process {os.getpid()}")
    set_start_method("spawn")
    processes = [Process(target=foo, args=(i, ))
                 for i in range(3)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("Done")
